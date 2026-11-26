"""
Zero-cost fake provider. Two modes:

* Default (no script) -- a tiny rule-based "agent" good enough for a
  keyless demo of the whole loop:
    - a user message starting with "!" becomes a bash tool call
      ("!sleep 20" is the easiest way to demo interrupt-during-tool),
    - after a tool result it summarizes the output,
    - otherwise it streams `canned_response` word by word, preceded by
      a short fake "thinking" stream.
* Scripted -- `script=[MockStep(...), ...]` is consumed one step per model
  call, for deterministic tests of tool loops, retries and errors.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..core.state import ConversationState, Message, TextBlock, ThinkingBlock, ToolUseBlock
from ..tools.base import ToolDefinition
from .base import (
    AgentProvider, MessageComplete, ProviderError, StreamEvent, TextDelta, ThinkingDelta,
    ToolCallEnd, ToolCallStart, UsageEvent,
)


@dataclass
class MockStep:
    text: str = ""
    thinking: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    error: str | None = None          # yield a ProviderError instead
    recoverable: bool = True
    error_after_words: int = 0         # stream this many words before failing


class MockProvider(AgentProvider):
    name = "mock"

    def __init__(
        self,
        canned_response: str = "This is a mock response from the zero-cost mock provider.",
        delay_s: float = 0.05,
        script: list[MockStep] | None = None,
        name: str = "mock",
    ):
        self.canned_response = canned_response
        self.delay_s = delay_s
        self.script = list(script) if script is not None else None
        self.name = name
        self.calls: list[ConversationState] = []  # snapshots, for test assertions

    def supports_interleaved_thinking(self, model: str | None = None) -> bool:
        return False

    def supports_prompt_caching(self) -> bool:
        return False

    def _next_step(self, state: ConversationState) -> MockStep:
        if self.script is not None:
            if not self.script:
                return MockStep(text="(mock script exhausted)")
            return self.script.pop(0)
        last = state.messages[-1] if state.messages else None
        if last is not None and last.role == "tool":
            out = " ".join(b.output for b in last.content if hasattr(b, "output"))
            snippet = out.strip().replace("\n", " ")[:200]
            return MockStep(text=f"The command finished. Output: {snippet}")
        if last is not None and last.role == "user":
            text = " ".join(b.text for b in last.content if isinstance(b, TextBlock)).strip()
            if ": !" in text:  # multiplayer attribution prefix, e.g. "alice: !ls"
                text = text.split(": ", 1)[1]
            if text.startswith("!"):
                return MockStep(thinking="The user wants me to run a command.",
                                tool_calls=[("bash", {"command": text[1:].strip()})])
        return MockStep(thinking="Let me think about this briefly.", text=self.canned_response)

    async def stream_turn(
        self, state: ConversationState, tools: list[ToolDefinition], model: str
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(state.model_copy(deep=True))
        step = self._next_step(state)
        blocks: list = []

        if step.thinking:
            for word in step.thinking.split(" "):
                await asyncio.sleep(self.delay_s / 2)
                yield ThinkingDelta(text=word + " ")
            blocks.append(ThinkingBlock(text=step.thinking, provider=self.name))

        if step.error:
            for i, word in enumerate(step.text.split(" ")[: step.error_after_words]):
                await asyncio.sleep(self.delay_s)
                yield TextDelta(text=word + " ")
            yield ProviderError(message=step.error, recoverable=step.recoverable)
            return

        text = ""
        if step.text:
            for word in step.text.split(" "):
                await asyncio.sleep(self.delay_s)  # cancellable checkpoint
                chunk = word + " "
                text += chunk
                yield TextDelta(text=chunk)
            blocks.append(TextBlock(text=text.strip()))

        for name, args in step.tool_calls:
            call = ToolUseBlock(tool_name=name, input=args)
            await asyncio.sleep(self.delay_s)
            yield ToolCallStart(id=call.id, tool_name=name)
            yield ToolCallEnd(id=call.id)
            blocks.append(call)

        yield UsageEvent(provider=self.name, model=model, input_tokens=0, output_tokens=len(text.split()))
        yield MessageComplete(message=Message(role="assistant", content=blocks, provider=self.name, model=model))
