from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..providers.base import AgentProvider, MessageComplete, StreamEvent, TextDelta, ThinkingDelta
from ..tools.base import ToolDefinition, ToolResult
from .interrupt import InterruptManager
from .state import ConversationState, LoopEvent, Message, TextBlock, ToolResultBlock, ToolUseBlock


class MaxIterationsExceeded(Exception):
    pass


_SENTINEL = object()
_CANCELLED = object()


class AgentLoop:
    """Orchestrates one conversation. Provider- and tool-agnostic: it only
    talks to the AgentProvider / ToolDefinition interfaces."""

    def __init__(
        self,
        state: ConversationState,
        providers: dict[str, AgentProvider],
        tools: dict[str, ToolDefinition],
    ):
        self.state = state
        self.providers = providers
        self.tools = tools
        self.interrupts = InterruptManager()

    def switch_provider(self, provider_name: str, model: str) -> None:
        """Mid-conversation provider/model switch. ConversationState is
        provider-agnostic, so this is just a pointer change -- the next
        stream_turn() call re-derives wire format from history."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        self.state.active_provider = provider_name
        self.state.active_model = model

    async def run_turn(self, user_input: str) -> AsyncIterator[StreamEvent]:
        """One full turn: user message -> (stream -> maybe tools -> stream)*
        -> done. Yields events for the caller (CLI/web) to render live.

        Cancellation model: the caller (CLI event loop) holds a reference
        to whatever task is running run_turn's consumption and can cancel
        IT directly -- see cli.py. That cancellation propagates down through
        this generator's `await` points (queue.get(), tool.execute()) via
        normal asyncio.CancelledError propagation. handle_interrupt() then
        does the state-machine bookkeeping and cleanup.
        """
        self.state.messages.append(
            Message(role="user", content=[TextBlock(text=user_input)])
        )
        self.state.transition(LoopEvent.USER_MESSAGE)
        self.state.iteration_count = 0

        while True:
            if self.state.iteration_count >= self.state.max_iterations:
                raise MaxIterationsExceeded(self.state.max_iterations)
            self.state.iteration_count += 1

            provider = self.providers[self.state.active_provider]
            async for event in self._stream_with_cancellation(provider):
                yield event
                if isinstance(event, MessageComplete):
                    # This is where completed turns actually enter history.
                    # Easy to forget -- caught by test_switch_provider_mid_conversation.
                    self.state.messages.append(event.message)

            pending_tool_calls = self._pending_tool_calls
            if not pending_tool_calls:
                self.state.transition(LoopEvent.STREAM_END)
                break

            self.state.transition(LoopEvent.TOOL_CALLS_REQUESTED)
            results = await self._run_tools(pending_tool_calls)
            self.state.messages.append(Message(role="tool", content=results))
            self.state.transition(LoopEvent.TOOL_CALLS_DONE)
            # loop continues: tool results are now in history for the
            # next stream_turn() call to pick up

    async def _stream_with_cancellation(self, provider: AgentProvider) -> AsyncIterator[StreamEvent]:
        """Bridges provider.stream_turn's async generator through a task +
        queue, so that a single well-defined asyncio.Task represents "the
        provider call" and can be tracked/cancelled by InterruptManager
        independently of how the caller is consuming events."""
        queue: asyncio.Queue = asyncio.Queue()
        self._pending_tool_calls: list[ToolUseBlock] = []
        accumulated_text = ""

        async def producer():
            try:
                async for event in provider.stream_turn(
                    self.state, list(self.tools.values()), self.state.active_model
                ):
                    if isinstance(event, MessageComplete):
                        self._pending_tool_calls = [
                            b for b in event.message.content if isinstance(b, ToolUseBlock)
                        ]
                    await queue.put(event)
                await queue.put(_SENTINEL)
            except asyncio.CancelledError:
                # IMPORTANT: a cancelled producer must NOT look like a
                # normal stream completion to the consumer below, or an
                # interrupt silently turns into "turn finished normally"
                # (this exact bug showed up in testing -- see README).
                await queue.put(_CANCELLED)
                raise

        task = asyncio.ensure_future(producer())
        self.interrupts.track(task)
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if item is _CANCELLED:
                    # Never let partial output vanish: whatever text
                    # streamed before the cut gets saved as a real
                    # message, explicitly flagged partial=True, so the
                    # next turn's context isn't silently missing it and
                    # the user can see what was cut off.
                    if accumulated_text:
                        self.state.messages.append(
                            Message(
                                role="assistant",
                                content=[TextBlock(text=accumulated_text)],
                                partial=True,
                            )
                        )
                    raise asyncio.CancelledError()
                if isinstance(item, (TextDelta, ThinkingDelta)):
                    accumulated_text += item.text
                yield item
        finally:
            self.interrupts.clear()
            if not task.done():
                task.cancel()

    async def _run_tools(self, calls: list[ToolUseBlock]) -> list[ToolResultBlock]:
        """Runs tool calls concurrently. Design decision (document in
        README): if interrupted mid-flight, in-flight calls are allowed to
        finish, but their result is reported to the model as cancelled so
        it doesn't assume a mutation was accepted by the user. Swap for
        hard-kill by cancelling `run_one`'s task instead if you prefer that
        tradeoff -- see BashTool.execute for the matching kill-on-cancel path."""
        async def run_one(call: ToolUseBlock) -> ToolResultBlock:
            tool = self.tools.get(call.tool_name)
            if tool is None:
                return ToolResultBlock(
                    tool_use_id=call.id, output=f"Unknown tool: {call.tool_name}", is_error=True
                )
            try:
                result: ToolResult = await tool.execute(**call.input)
                return ToolResultBlock(tool_use_id=call.id, output=result.output, is_error=result.is_error)
            except asyncio.CancelledError:
                return ToolResultBlock(tool_use_id=call.id, output="Cancelled by user", is_error=True)

        return await asyncio.gather(*(run_one(c) for c in calls))

    async def handle_interrupt(self) -> None:
        """Call this from the CLI/web layer when the user interrupts.
        Cancels whatever's in flight (partial text is captured as a
        partial=True message inside _stream_with_cancellation before the
        cancellation propagates) and returns the loop to IDLE."""
        was_running = await self.interrupts.interrupt()
        if was_running:
            self.state.transition(LoopEvent.INTERRUPT_REQUESTED)
        self.state.transition(LoopEvent.CLEANUP_DONE)
