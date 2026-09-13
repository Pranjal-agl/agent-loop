from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..core.state import ConversationState, Message, TextBlock
from ..tools.base import ToolDefinition
from .base import AgentProvider, MessageComplete, StreamEvent, TextDelta


class MockProvider(AgentProvider):
    """No network calls. Fakes a slow token-by-token stream so interrupt
    handling and CLI rendering can be built/tested with zero API cost.
    canned_response can be swapped per-test to simulate tool calls, errors,
    long loops, etc."""

    name = "mock"

    def __init__(self, canned_response: str = "This is a mock response.", delay_s: float = 0.05):
        self.canned_response = canned_response
        self.delay_s = delay_s

    def supports_interleaved_thinking(self) -> bool:
        return False

    def supports_prompt_caching(self) -> bool:
        return False

    def build_cache_control(self, state: ConversationState) -> dict:
        return {}

    async def stream_turn(
        self, state: ConversationState, tools: list[ToolDefinition], model: str
    ) -> AsyncIterator[StreamEvent]:
        text = ""
        for word in self.canned_response.split(" "):
            await asyncio.sleep(self.delay_s)  # cancellable checkpoint
            chunk = word + " "
            text += chunk
            yield TextDelta(text=chunk)
        yield MessageComplete(
            message=Message(role="assistant", content=[TextBlock(text=text.strip())])
        )
