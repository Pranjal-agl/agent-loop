"""
The AgentProvider protocol is the entire multi-provider abstraction.
Adding a provider = implementing one class against this interface
(GeminiProvider is the proof: it touches nothing outside its own file).

Design note: stream_turn yields StreamEvents rather than returning a
final Message, because the loop needs to observe partial state (for
interrupt handling, for updating a live CLI/UI) as it arrives.
"""
from __future__ import annotations

import json
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


class UsageEvent(BaseModel):
    """Token accounting for one model call. cache_* fields are how the
    demo shows prompt caching actually working."""
    type: Literal["usage"] = "usage"
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


class Notice(BaseModel):
    """Non-fatal, human-readable information (e.g. 'thinking disabled for
    this request because the history came from another provider')."""
    type: Literal["notice"] = "notice"
    text: str


class MessageComplete(BaseModel):
    """Terminal event for a model call: carries the fully-assembled
    Message so the loop can append it to ConversationState.messages."""
    type: Literal["message_complete"] = "message_complete"
    message: Message


class ProviderError(BaseModel):
    type: Literal["error"] = "error"
    message: str
    # True for transient failures (rate limit, overloaded, 5xx, network):
    # the loop retries these with backoff. Nothing is committed to history
    # until MessageComplete, so a retry is always state-safe.
    recoverable: bool = False


StreamEvent = Union[
    TextDelta, ThinkingDelta, ToolCallStart, ToolCallArgsDelta,
    ToolCallEnd, UsageEvent, Notice, MessageComplete, ProviderError,
]


def error_from_exception(exc: BaseException) -> ProviderError:
    """Classify an SDK exception as transient (retry) or fatal, without
    importing any SDK: all three SDKs expose `status_code` on HTTP errors
    and name their network errors *ConnectionError / *TimeoutError."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    name = type(exc).__name__
    text = str(exc) or name
    recoverable = (
        (isinstance(status, int) and (status in (408, 409, 429) or status >= 500))
        or "Connection" in name
        or "Timeout" in name
        or "overloaded" in text.lower()
    )
    return ProviderError(message=f"{name}: {text}", recoverable=recoverable)


def parse_tool_arguments(raw: str | None) -> dict:
    """Models occasionally emit invalid JSON arguments. Instead of crashing
    the adapter, carry the raw string; the loop reports it back to the
    model as a tool error so it can retry the call."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"__raw_arguments__": raw}
    except json.JSONDecodeError:
        return {"__raw_arguments__": raw}


class AgentProvider(ABC):
    name: str

    @abstractmethod
    def supports_interleaved_thinking(self, model: str | None = None) -> bool: ...

    @abstractmethod
    def supports_prompt_caching(self) -> bool: ...

    @abstractmethod
    def stream_turn(
        self,
        state: ConversationState,
        tools: list[ToolDefinition],
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        """Convert `state` to this provider's wire format, call the API
        with streaming enabled, and yield StreamEvents as they arrive,
        ending with exactly one MessageComplete or ProviderError.

        Contract:
          * Cancelling the consuming task must tear down the HTTP stream
            (use the SDK's async context manager so __aexit__ runs).
          * Never catch asyncio.CancelledError (only `Exception`).
          * Must accept *any* valid ConversationState, including history
            produced by other providers and histories left behind by an
            interrupt. Repairing provider-specific invariants (dangling
            tool calls, foreign reasoning blocks, role alternation) is the
            adapter's job, not the loop's.
        """
        ...

    def build_cache_control(self, state: ConversationState) -> dict:
        """Describe provider-specific cache markers (introspection only).
        No-op for providers without explicit prompt caching."""
        return {}
