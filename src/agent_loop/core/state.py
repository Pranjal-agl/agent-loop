"""
Provider-agnostic conversation state.

Every provider adapter converts ConversationState <-> its own wire format.
Nothing outside providers/ should ever import an OpenAI or Anthropic type.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Loop state machine
# --------------------------------------------------------------------------

class LoopState(str, Enum):
    IDLE = "idle"                  # waiting for user input
    STREAMING = "streaming"        # provider is streaming a response
    TOOL_RUNNING = "tool_running"  # executing one or more tool calls
    INTERRUPTED = "interrupted"    # user cancelled; cleanup in progress
    DONE = "done"                  # turn complete, back to idle next tick
    ERROR = "error"                # unrecoverable error this turn


class LoopEvent(str, Enum):
    USER_MESSAGE = "user_message"
    STREAM_START = "stream_start"
    STREAM_END = "stream_end"
    TOOL_CALLS_REQUESTED = "tool_calls_requested"
    TOOL_CALLS_DONE = "tool_calls_done"
    INTERRUPT_REQUESTED = "interrupt_requested"
    CLEANUP_DONE = "cleanup_done"
    ERROR_RAISED = "error_raised"


# Explicit transition table -- the state machine is data, not scattered
# if/else. Any (state, event) pair not listed here is invalid and raises.
VALID_TRANSITIONS: dict[tuple[LoopState, LoopEvent], LoopState] = {
    (LoopState.IDLE, LoopEvent.USER_MESSAGE): LoopState.STREAMING,
    (LoopState.STREAMING, LoopEvent.TOOL_CALLS_REQUESTED): LoopState.TOOL_RUNNING,
    (LoopState.STREAMING, LoopEvent.STREAM_END): LoopState.DONE,
    (LoopState.TOOL_RUNNING, LoopEvent.TOOL_CALLS_DONE): LoopState.STREAMING,
    (LoopState.STREAMING, LoopEvent.INTERRUPT_REQUESTED): LoopState.INTERRUPTED,
    (LoopState.TOOL_RUNNING, LoopEvent.INTERRUPT_REQUESTED): LoopState.INTERRUPTED,
    (LoopState.INTERRUPTED, LoopEvent.CLEANUP_DONE): LoopState.IDLE,
    (LoopState.DONE, LoopEvent.USER_MESSAGE): LoopState.STREAMING,
    (LoopState.STREAMING, LoopEvent.ERROR_RAISED): LoopState.ERROR,
    (LoopState.TOOL_RUNNING, LoopEvent.ERROR_RAISED): LoopState.ERROR,
    (LoopState.ERROR, LoopEvent.CLEANUP_DONE): LoopState.IDLE,
    # NOTE for implementer: decide here whether TOOL_RUNNING -> INTERRUPTED
    # lets the in-flight tool call finish (recommended: yes, then discard
    # its result) or hard-kills it. Document the choice in the README and
    # make sure this table + the InterruptManager agree with each other.
}


# --------------------------------------------------------------------------
# Content blocks (mirrors the union of what OpenAI + Claude can emit)
# --------------------------------------------------------------------------

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """Claude extended-thinking / interleaved-thinking content.
    Providers that don't support this simply never produce it."""
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None  # Claude's opaque signature, if provided


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_name: str
    input: dict


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    output: str
    is_error: bool = False


ContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock]


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: list[ContentBlock]
    created_at: float = Field(default_factory=time.time)
    # Marks a message that was cut short by an interrupt -- kept in
    # history so the model has context, but adapters may need to
    # special-case it when converting to wire format (e.g. Claude
    # requires the last assistant turn to end cleanly before a new
    # user turn; OpenAI is more permissive).
    partial: bool = False


class CacheBreakpoint(BaseModel):
    """Where to insert an explicit Anthropic prompt-cache marker.
    `after_message_index` is an index into ConversationState.messages."""
    after_message_index: int
    label: str = ""


class ConversationState(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[Message] = Field(default_factory=list)
    state: LoopState = LoopState.IDLE
    active_provider: str = "mock"
    active_model: str = "mock-1"
    iteration_count: int = 0
    max_iterations: int = 15
    cache_breakpoints: list[CacheBreakpoint] = Field(default_factory=list)

    def transition(self, event: LoopEvent) -> LoopState:
        key = (self.state, event)
        if key not in VALID_TRANSITIONS:
            raise ValueError(f"Invalid transition: {self.state} + {event}")
        self.state = VALID_TRANSITIONS[key]
        return self.state
