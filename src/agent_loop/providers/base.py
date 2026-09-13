"""
The AgentProvider protocol is the entire multi-provider abstraction.
Adding Gemini later = implementing one new class against this interface.

Design note: stream_turn yields StreamEvents rather than returning a
final Message, because the loop needs to observe partial state (for
interrupt handling, for updating a live CLI/UI) as it arrives -- it
can't just await a single coroutine to completion.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Literal, Union

from pydantic import BaseModel

from ..core.state import ConversationState, Message
from ..tools.base import ToolDefinition


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class ToolCallStart(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    id: str
    tool_name: str


class ToolCallArgsDelta(BaseModel):
    type: Literal["tool_call_args_delta"] = "tool_call_args_delta"
    id: str
    partial_json: str


class ToolCallEnd(BaseModel):
    type: Literal["tool_call_end"] = "tool_call_end"
    id: str


class MessageComplete(BaseModel):
    """Terminal event for a turn: carries the fully-assembled Message
    so the loop can append it to ConversationState.messages verbatim."""
    type: Literal["message_complete"] = "message_complete"
    message: Message


class ProviderError(BaseModel):
    type: Literal["error"] = "error"
    message: str
    recoverable: bool = False


StreamEvent = Union[
    TextDelta, ThinkingDelta, ToolCallStart, ToolCallArgsDelta,
    ToolCallEnd, MessageComplete, ProviderError,
]


class AgentProvider(ABC):
    name: str

    @abstractmethod
    def supports_interleaved_thinking(self) -> bool: ...

    @abstractmethod
    def supports_prompt_caching(self) -> bool: ...

    @abstractmethod
    async def stream_turn(
        self,
        state: ConversationState,
        tools: list[ToolDefinition],
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        """Convert `state` to this provider's wire format, call the API
        with streaming enabled, and yield StreamEvents as they arrive.

        MUST be written so that cancelling the enclosing asyncio.Task
        (via task.cancel()) cleanly tears down the underlying HTTP
        stream -- e.g. by using the SDK's async streaming context
        manager so __aexit__ runs on CancelledError. This is what makes
        mid-stream interruption possible.

        Implementer note -- resume semantics differ by provider and
        must be handled here, not in the shared loop:
          * OpenAI: a partial assistant message with no matching tool
            result is generally fine to just drop from history.
          * Claude: if a tool_use block was emitted before interruption,
            Claude expects a matching tool_result (even a synthetic
            "cancelled" one) before the next turn, or the API call will
            be rejected. Handle that reconciliation here.
        """
        ...

    @abstractmethod
    def build_cache_control(self, state: ConversationState) -> dict:
        """Return provider-specific cache-control markers derived from
        state.cache_breakpoints. No-op (return {}) for providers that
        don't support explicit prompt caching."""
        ...
