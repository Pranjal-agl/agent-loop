"""
Events emitted by the AgentLoop (on top of the provider StreamEvents it
forwards) and by Session (multiplayer). Every event is a pydantic model
with a `type` discriminator so the web server can JSON-encode them as-is.
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel

from ..providers.base import StreamEvent


class ToolExecutionStart(BaseModel):
    type: Literal["tool_exec_start"] = "tool_exec_start"
    id: str
    tool_name: str
    input: dict


class ToolExecutionEnd(BaseModel):
    type: Literal["tool_exec_end"] = "tool_exec_end"
    id: str
    tool_name: str
    output: str
    is_error: bool
    duration_s: float


class Retrying(BaseModel):
    type: Literal["retrying"] = "retrying"
    attempt: int
    delay_s: float
    error: str
    # If text already streamed for the failed attempt, UIs should
    # visually discard it -- it is not in history.
    discarded_partial: bool


class TurnEnded(BaseModel):
    type: Literal["turn_ended"] = "turn_ended"
    reason: Literal["completed", "max_iterations", "error", "interrupted"]
    iterations: int = 0
    detail: str = ""


class InterruptSummary(BaseModel):
    """What the loop did with in-flight work when it was interrupted.
    This is what gets shown to the user ("kept 112 chars of partial
    text; bash killed after 3.2s") -- see AgentLoop._reconcile."""
    type: Literal["interrupt_summary"] = "interrupt_summary"
    phase: str
    kept_partial_text_chars: int = 0
    discarded: list[str] = []
    completed_tools: list[str] = []
    killed_tools: list[str] = []
    skipped_tools: list[str] = []

    def describe(self) -> str:
        parts = [f"interrupted during {self.phase}"]
        if self.kept_partial_text_chars:
            parts.append(f"kept {self.kept_partial_text_chars} chars of partial text in context")
        if self.completed_tools:
            parts.append("completed: " + ", ".join(self.completed_tools))
        if self.killed_tools:
            parts.append("killed: " + ", ".join(self.killed_tools))
        if self.skipped_tools:
            parts.append("not run: " + ", ".join(self.skipped_tools))
        if self.discarded:
            parts.append("discarded: " + ", ".join(self.discarded))
        return "; ".join(parts)


# ---- Session-level (multiplayer) events ---------------------------------

class UserMessageEvent(BaseModel):
    type: Literal["user_message"] = "user_message"
    author: str
    text: str
    queued: bool = False  # True if it is waiting behind a running turn


class TurnStarted(BaseModel):
    type: Literal["turn_started"] = "turn_started"
    authors: list[str]
    provider: str
    model: str


class TurnInterrupted(BaseModel):
    type: Literal["turn_interrupted"] = "turn_interrupted"
    by: str
    summary: InterruptSummary | None = None


class ProviderSwitched(BaseModel):
    type: Literal["provider_switched"] = "provider_switched"
    by: str
    provider: str
    model: str


class Presence(BaseModel):
    type: Literal["presence"] = "presence"
    users: list[str]
    joined: str | None = None
    left: str | None = None


class Snapshot(BaseModel):
    """Sent to a client when it attaches, so late joiners see history."""
    type: Literal["snapshot"] = "snapshot"
    session_id: str
    state: dict[str, Any]
    providers: list[str]
    users: list[str]
    queued: int
    # alias -> (provider, model), so UIs can offer a model picker
    models: dict[str, tuple[str, str]] = {}


LoopOutput = Union[StreamEvent, ToolExecutionStart, ToolExecutionEnd, Retrying, TurnEnded]
