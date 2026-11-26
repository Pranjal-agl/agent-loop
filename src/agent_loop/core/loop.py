"""
The agent loop: user message -> (model call -> tool calls)* -> done.

Interrupt model (the core of this assignment)
---------------------------------------------
There is exactly ONE cancellation mechanism: cancel the asyncio.Task that
is iterating `run_turn()`. Wherever the turn happens to be suspended --
reading the model's HTTP stream, waiting on a subprocess, sleeping in a
retry backoff, or sitting between two iterations -- CancelledError lands
at that await, and `run_turn` catches it in one place and calls
`_reconcile()`, which uses the per-turn `_TurnContext` to know exactly
what was in flight and writes a *structurally valid* history:

  * streamed-but-uncommitted assistant text  -> kept as a partial=True
    assistant message (the user saw it, so the model should too)
  * streamed thinking / half-streamed tool-call args -> discarded (no
    signature yet, never executed; not replayable to any provider)
  * a committed assistant message whose tool calls have no results yet
    -> every call gets exactly one result: its real output if it finished,
    its partial output + "killed" if it was running, "not run" otherwise
  * a synthetic user note telling the model it was interrupted

The state machine then goes STREAMING/TOOL_RUNNING -> INTERRUPTED -> IDLE,
so the next user message (on any provider) starts from a clean state.
`AgentLoop.interrupt()` is a convenience that cancels the tracked task.
"""
from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..providers.base import (
    AgentProvider, MessageComplete, ProviderError, TextDelta, ThinkingDelta, ToolCallStart,
)
from ..tools.base import ToolDefinition, ToolResult, truncate_output
from .events import (
    InterruptSummary, LoopOutput, Retrying, ToolExecutionEnd, ToolExecutionStart, TurnEnded,
)
from .state import (
    BUSY_STATES, ConversationState, LoopEvent, LoopState, Message, TextBlock, ToolResultBlock,
    ToolUseBlock,
)

INTERRUPT_NOTE = (
    "[The user interrupted the previous response{where}. Anything above that was cut off "
    "is incomplete. Do not resume the interrupted work unless the user asks you to; "
    "respond to the user's next message.]"
)


@dataclass
class _TurnContext:
    """Everything that is in flight but not yet committed to history.
    This is what makes reconciliation precise instead of guesswork."""
    iteration: int = 0
    # current model call
    stream_open: bool = False
    partial_text: str = ""
    partial_thinking_chars: int = 0
    partial_tool_calls: int = 0
    completed: Message | None = None
    error: ProviderError | None = None
    # current tool phase: calls from the committed assistant message
    pending_calls: list[ToolUseBlock] = field(default_factory=list)
    results: dict[str, ToolResultBlock] = field(default_factory=dict)
    running: dict[str, tuple[asyncio.Task, float]] = field(default_factory=dict)

    def begin_stream(self) -> None:
        self.stream_open = True
        self.partial_text = ""
        self.partial_thinking_chars = 0
        self.partial_tool_calls = 0
        self.completed = None
        self.error = None


class AgentLoop:
    """Orchestrates one conversation. Provider- and tool-agnostic: it only
    talks to the AgentProvider / ToolDefinition interfaces."""

    def __init__(
        self,
        state: ConversationState,
        providers: dict[str, AgentProvider],
        tools: dict[str, ToolDefinition],
        *,
        max_retries: int = 3,
        retry_base_delay_s: float = 1.0,
        max_tool_output_chars: int = 30_000,
    ):
        self.state = state
        self.providers = providers
        self.tools = tools
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.max_tool_output_chars = max_tool_output_chars
        self.last_interrupt: InterruptSummary | None = None
        self._turn_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ API

    def switch_provider(self, provider_name: str, model: str) -> None:
        """Switch provider/model. ConversationState is provider-agnostic,
        so this is a pointer change; the next model call re-derives the
        wire format from history (dropping reasoning artifacts the new
        provider can't read). If called mid-turn it takes effect at the
        next model call of the current turn."""
        if provider_name not in self.providers:
            raise ValueError(
                f"Unknown provider: {provider_name} (available: {', '.join(self.providers)})"
            )
        self.state.active_provider = provider_name
        self.state.active_model = model

    async def interrupt(self) -> bool:
        """Cancel the running turn (if any) and wait until reconciliation
        has finished. Idempotent: returns False if nothing was running,
        so two users pressing Stop at once is harmless."""
        task = self._turn_task
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.wait([task])
        return True

    # Back-compat name used by older callers/tests.
    handle_interrupt = interrupt

    async def run_turn(self, user_input: str, author: str | None = None) -> AsyncIterator[LoopOutput]:
        """Run one full turn and yield events for the caller to render.
        To make `interrupt()` work, iterate this from a task (Session and
        the CLI do); cancelling that task from anywhere also works."""
        if self.state.busy:
            raise RuntimeError("A turn is already running; interrupt it or wait.")
        self._turn_task = asyncio.current_task()
        self.last_interrupt = None
        self.state.messages.append(
            Message(role="user", content=[TextBlock(text=user_input)], author=author)
        )
        self.state.transition(LoopEvent.USER_MESSAGE)
        self.state.iteration_count = 0
        ctx = _TurnContext()

        try:
            while True:
                if ctx.iteration >= self.state.max_iterations:
                    # Not an exception: the history is valid (it ends with
                    # tool results), so the user can simply say "continue".
                    self.state.transition(LoopEvent.STREAM_END)
                    yield TurnEnded(
                        reason="max_iterations", iterations=ctx.iteration,
                        detail=f"Stopped after {ctx.iteration} model calls; send 'continue' to keep going.",
                    )
                    return
                ctx.iteration += 1
                self.state.iteration_count = ctx.iteration

                async for event in self._call_model(ctx):
                    yield event

                if ctx.error is not None:
                    self.state.transition(LoopEvent.ERROR_RAISED)
                    self.state.transition(LoopEvent.CLEANUP_DONE)
                    yield TurnEnded(reason="error", iterations=ctx.iteration, detail=ctx.error.message)
                    return

                message = ctx.completed
                assert message is not None
                calls = [b for b in message.content if isinstance(b, ToolUseBlock)]
                # Commit the assistant message and the "tools pending" fact
                # together, with no await in between, so reconciliation
                # never sees one without the other.
                self.state.messages.append(message)
                ctx.completed = None
                if not calls:
                    self.state.transition(LoopEvent.STREAM_END)
                    yield TurnEnded(reason="completed", iterations=ctx.iteration)
                    return

                ctx.pending_calls = calls
                ctx.results = {}
                self.state.transition(LoopEvent.TOOL_CALLS_REQUESTED)
                async for event in self._run_tools(ctx):
                    yield event
                self.state.messages.append(
                    Message(role="tool", content=[ctx.results[c.id] for c in calls])
                )
                ctx.pending_calls = []
                self.state.transition(LoopEvent.TOOL_CALLS_DONE)

        except (asyncio.CancelledError, GeneratorExit):
            await self._reconcile(ctx, reason="interrupt")
            raise
        except Exception as exc:  # a bug or tool crash -- never leave state wedged
            await self._reconcile(ctx, reason="error")
            yield TurnEnded(reason="error", iterations=ctx.iteration, detail=f"{type(exc).__name__}: {exc}")
        finally:
            if self._turn_task is asyncio.current_task():
                self._turn_task = None

    # -------------------------------------------------------------- model

    async def _call_model(self, ctx: _TurnContext) -> AsyncIterator[LoopOutput]:
        """One model call, with retries for transient errors. Nothing is
        committed to state here; the caller commits ctx.completed."""
        for attempt in range(self.max_retries + 1):
            provider = self.providers[self.state.active_provider]
            model = self.state.active_model
            ctx.begin_stream()
            error: ProviderError | None = None
            stream = provider.stream_turn(self.state, list(self.tools.values()), model)
            async with contextlib.aclosing(stream):
                async for event in stream:
                    if isinstance(event, ProviderError):
                        error = event
                        break
                    if isinstance(event, TextDelta):
                        ctx.partial_text += event.text
                    elif isinstance(event, ThinkingDelta):
                        ctx.partial_thinking_chars += len(event.text)
                    elif isinstance(event, ToolCallStart):
                        ctx.partial_tool_calls += 1
                    elif isinstance(event, MessageComplete):
                        event.message.provider = event.message.provider or provider.name
                        event.message.model = event.message.model or model
                        ctx.completed = event.message
                    yield event

            if error is None and ctx.completed is not None:
                ctx.stream_open = False
                return
            if error is None:
                error = ProviderError(message="Provider stream ended without a final message", recoverable=True)

            if error.recoverable and attempt < self.max_retries:
                delay = self.retry_base_delay_s * (2 ** attempt) * (0.75 + random.random() / 2)
                discarded = bool(ctx.partial_text or ctx.partial_tool_calls)
                ctx.stream_open = False  # the failed attempt's partial output is void
                yield Retrying(
                    attempt=attempt + 1, delay_s=round(delay, 2), error=error.message,
                    discarded_partial=discarded,
                )
                await asyncio.sleep(delay)  # cancellable: interrupt during backoff is fine
                continue

            ctx.stream_open = False
            ctx.error = error
            yield error
            return

    # -------------------------------------------------------------- tools

    async def _run_tools(self, ctx: _TurnContext) -> AsyncIterator[LoopOutput]:
        """Executes tool calls sequentially, in the order the model emitted
        them. Sequential (rather than gather) is deliberate: side effects
        happen in a deterministic order, and an interrupt leaves every call
        in exactly one of three states -- completed, killed, or not run --
        which is what lets us tell the model precisely what happened."""
        for call in ctx.pending_calls:
            yield ToolExecutionStart(id=call.id, tool_name=call.tool_name, input=call.input)
            started = time.monotonic()
            task = asyncio.ensure_future(self._execute_one(call))
            ctx.running[call.id] = (task, started)
            try:
                # shield: cancelling *us* must not implicitly cancel the
                # tool; _reconcile cancels it explicitly so it can still
                # collect the tool's partial output.
                result = await asyncio.shield(task)
            finally:
                if task.done():
                    ctx.running.pop(call.id, None)
            ctx.results[call.id] = result
            yield ToolExecutionEnd(
                id=call.id, tool_name=call.tool_name, output=result.output,
                is_error=result.is_error, duration_s=round(time.monotonic() - started, 3),
            )

    async def _execute_one(self, call: ToolUseBlock) -> ToolResultBlock:
        tool = self.tools.get(call.tool_name)
        if tool is None:
            return ToolResultBlock(
                tool_use_id=call.id, tool_name=call.tool_name, is_error=True,
                output=f"Unknown tool: {call.tool_name}. Available: {', '.join(self.tools)}",
            )
        if "__raw_arguments__" in call.input:
            return ToolResultBlock(
                tool_use_id=call.id, tool_name=call.tool_name, is_error=True,
                output="Tool arguments were not valid JSON: " + str(call.input["__raw_arguments__"])[:500],
            )
        try:
            result: ToolResult = await tool.execute(**call.input)
        except TypeError as exc:  # wrong/missing arguments from the model
            result = ToolResult(output=f"Invalid arguments for {call.tool_name}: {exc}", is_error=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a tool bug must not take down the loop
            result = ToolResult(output=f"{call.tool_name} failed: {type(exc).__name__}: {exc}", is_error=True)
        return ToolResultBlock(
            tool_use_id=call.id, tool_name=call.tool_name, is_error=result.is_error,
            output=truncate_output(result.output or "(no output)", self.max_tool_output_chars),
        )

    # ------------------------------------------------------ reconciliation

    async def _reconcile(self, ctx: _TurnContext, reason: str) -> None:
        """Bring history + state machine back to a valid resting state
        after an interrupt (or unexpected error) at any point in a turn."""
        if self.state.state not in BUSY_STATES:
            return  # interrupted after the turn already finished: nothing to do
        phase = "tool execution" if self.state.state == LoopState.TOOL_RUNNING else "model streaming"
        if ctx.running:
            phase = "tool execution (" + ", ".join(
                next(c.tool_name for c in ctx.pending_calls if c.id == cid) for cid in ctx.running
            ) + ")"
        summary = InterruptSummary(phase=phase)

        # 1. In-flight model output.
        if ctx.stream_open:
            text = ctx.partial_text.rstrip()
            if text:
                self.state.messages.append(Message(
                    role="assistant", content=[TextBlock(text=text)], partial=True,
                    provider=self.state.active_provider, model=self.state.active_model,
                ))
                summary.kept_partial_text_chars = len(text)
            if ctx.partial_thinking_chars:
                summary.discarded.append(f"{ctx.partial_thinking_chars} chars of unsigned partial thinking")
            if ctx.partial_tool_calls:
                summary.discarded.append(f"{ctx.partial_tool_calls} half-streamed tool call(s) (never executed)")
        elif ctx.completed is not None:
            # Fully received but not yet committed (cancel landed exactly
            # between the stream ending and the commit). Keep it -- but
            # only its text: committing un-run tool calls would need results.
            texts = [b for b in ctx.completed.content if isinstance(b, TextBlock) and b.text.strip()]
            if texts:
                self.state.messages.append(Message(
                    role="assistant", content=texts, partial=True,
                    provider=ctx.completed.provider, model=ctx.completed.model,
                ))

        # 2. Tool calls from the committed assistant message that have no
        #    results yet. Every call gets exactly one result.
        if ctx.pending_calls:
            running = dict(ctx.running)
            for task, _ in running.values():
                task.cancel()  # tools kill their subprocess and return partial output
            if running:
                await asyncio.wait([t for t, _ in running.values()], timeout=5)
            results: list[ToolResultBlock] = []
            for call in ctx.pending_calls:
                label = f"{call.tool_name}#{call.id[-6:]}"
                if call.id in ctx.results:
                    results.append(ctx.results[call.id])
                    summary.completed_tools.append(label)
                elif call.id in running:
                    task, started = running[call.id]
                    elapsed = time.monotonic() - started
                    partial = ""
                    if task.done() and not task.cancelled() and task.exception() is None:
                        partial = task.result().output
                    results.append(ToolResultBlock(
                        tool_use_id=call.id, tool_name=call.tool_name, is_error=True,
                        output=(
                            f"Interrupted by the user after {elapsed:.1f}s; the command was killed "
                            f"and any side effects may be partial.\n{partial}"
                        ).strip(),
                    ))
                    summary.killed_tools.append(f"{label} after {elapsed:.1f}s")
                else:
                    results.append(ToolResultBlock(
                        tool_use_id=call.id, tool_name=call.tool_name, is_error=True,
                        output="Not executed: the user interrupted before this tool call ran.",
                    ))
                    summary.skipped_tools.append(label)
            self.state.messages.append(Message(role="tool", content=results))
            ctx.pending_calls = []

        # 3. Tell the model what happened, then reset the state machine.
        if reason == "interrupt":
            where = " while a tool was running" if summary.killed_tools else ""
            self.state.messages.append(Message(
                role="user", synthetic=True,
                content=[TextBlock(text=INTERRUPT_NOTE.format(where=where))],
            ))
            self.state.transition(LoopEvent.INTERRUPT_REQUESTED)
            self.last_interrupt = summary
        else:
            self.state.transition(LoopEvent.ERROR_RAISED)
        self.state.transition(LoopEvent.CLEANUP_DONE)

