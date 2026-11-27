"""
Provider-agnostic conversation state.

Every provider adapter converts ConversationState <-> its own wire format.
Nothing outside providers/ should ever import an OpenAI, Anthropic or
Gemini type. Provider-specific opaque data (Claude thinking signatures,
OpenAI encrypted reasoning, Gemini thought signatures) rides along in
`provider_meta` / `signature` fields that other providers simply ignore.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Loop state machine
# --------------------------------------------------------------------------

class LoopState(str, Enum):
    IDLE = "idle"                  # waiting for user input
    STREAMING = "streaming"        # provider is streaming a response
    TOOL_RUNNING = "tool_running"  # executing tool calls
    INTERRUPTED = "interrupted"    # user cancelled; reconciliation in progress
    DONE = "done"                  # turn complete (equivalent to IDLE for new input)
    ERROR = "error"                # unrecoverable error this turn; cleanup in progress


class LoopEvent(str, Enum):
    USER_MESSAGE = "user_message"
    STREAM_END = "stream_end"
    TOOL_CALLS_REQUESTED = "tool_calls_requested"
    TOOL_CALLS_DONE = "tool_calls_done"
    INTERRUPT_REQUESTED = "interrupt_requested"
    CLEANUP_DONE = "cleanup_done"
    ERROR_RAISED = "error_raised"


# The state machine is data, not scattered if/else. Any (state, event)
# pair not listed here is invalid and raises -- which is exactly what
# surfaced several of the interrupt bugs described in the README.
VALID_TRANSITIONS: dict[tuple[LoopState, LoopEvent], LoopState] = {
    (LoopState.IDLE, LoopEvent.USER_MESSAGE): LoopState.STREAMING,
    (LoopState.DONE, LoopEvent.USER_MESSAGE): LoopState.STREAMING,
    (LoopState.STREAMING, LoopEvent.TOOL_CALLS_REQUESTED): LoopState.TOOL_RUNNING,
    (LoopState.STREAMING, LoopEvent.STREAM_END): LoopState.DONE,
    (LoopState.TOOL_RUNNING, LoopEvent.TOOL_CALLS_DONE): LoopState.STREAMING,
    # Interrupts are accepted from both "busy" states. TOOL_RUNNING ->
    # INTERRUPTED kills in-flight tools (see AgentLoop._reconcile); the
    # tool's partial output is still reported to the model.
    (LoopState.STREAMING, LoopEvent.INTERRUPT_REQUESTED): LoopState.INTERRUPTED,
    (LoopState.TOOL_RUNNING, LoopEvent.INTERRUPT_REQUESTED): LoopState.INTERRUPTED,
    (LoopState.INTERRUPTED, LoopEvent.CLEANUP_DONE): LoopState.IDLE,
    (LoopState.STREAMING, LoopEvent.ERROR_RAISED): LoopState.ERROR,
    (LoopState.TOOL_RUNNING, LoopEvent.ERROR_RAISED): LoopState.ERROR,
    (LoopState.ERROR, LoopEvent.CLEANUP_DONE): LoopState.IDLE,
}

BUSY_STATES = {LoopState.STREAMING, LoopState.TOOL_RUNNING}


# --------------------------------------------------------------------------
# Content blocks (the union of what OpenAI, Claude and Gemini can emit)
# --------------------------------------------------------------------------

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    # Opaque provider data keyed by provider name, e.g.
    # {"gemini": {"thought_signature": "..."}}.
    provider_meta: dict[str, Any] = Field(default_factory=dict)


class ThinkingBlock(BaseModel):
    """Model reasoning. `provider` records who produced it, because
    reasoning artifacts are only valid when replayed to the provider that
    produced them (a Claude thinking signature means nothing to OpenAI and
    vice versa). Adapters drop foreign ThinkingBlocks on conversion."""
    type: Literal["thinking"] = "thinking"
    text: str = ""
    provider: str | None = None
    # Claude: thinking signature, or the encrypted `data` of a
    # redacted_thinking block when redacted=True.
    signature: str | None = None
    redacted: bool = False
    # Other providers' opaque data, e.g. OpenAI {"id", "encrypted_content"}.
    provider_meta: dict[str, Any] = Field(default_factory=dict)


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])
    tool_name: str
    input: dict
    provider_meta: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    output: str
    is_error: bool = False
    tool_name: str | None = None  # needed by Gemini, handy for UIs


ContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock]


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: list[ContentBlock]
    created_at: float = Field(default_factory=time.time)
    # Cut short by an interrupt. Kept in history so the model sees what
    # the user saw, but only *text* is ever kept in a partial message
    # (partial thinking has no signature and can't be replayed).
    partial: bool = False
    # Written by the loop itself (e.g. "the user interrupted"), not typed
    # by a human. UIs can render these differently.
    synthetic: bool = False
    # Multiplayer attribution: who sent a user message.
    author: str | None = None
    # Which provider/model produced an assistant message.
    provider: str | None = None
    model: str | None = None


class CacheBreakpoint(BaseModel):
    """An extra, user-pinned Anthropic prompt-cache marker.
    `after_message_index` is an index into ConversationState.messages.
    (The Anthropic adapter already places automatic breakpoints on the
    system prompt and the moving tail of the conversation.)"""
    after_message_index: int
    label: str = ""


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful coding agent working in a local project directory. "
    "You have three tools: file_read, file_edit and bash. Prefer reading files "
    "before editing them, make small targeted edits, and verify your work by "
    "running commands. Paths are relative to the working directory. "
    "If a message says the user interrupted you, stop what you were doing and "
    "focus on the user's newest message."
)


class ConversationState(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    messages: list[Message] = Field(default_factory=list)
    state: LoopState = LoopState.IDLE
    active_provider: str = "mock"
    active_model: str = "mock-1"
    iteration_count: int = 0
    max_iterations: int = 50
    cache_breakpoints: list[CacheBreakpoint] = Field(default_factory=list)

    def transition(self, event: LoopEvent) -> LoopState:
        key = (self.state, event)
        if key not in VALID_TRANSITIONS:
            raise ValueError(f"Invalid transition: {self.state.value} + {event.value}")
        self.state = VALID_TRANSITIONS[key]
        return self.state

    @property
    def busy(self) -> bool:
        return self.state in BUSY_STATES
