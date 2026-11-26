"""
Claude adapter (Messages API, streaming).

Provider-specific behaviour implemented here
--------------------------------------------
Extended / interleaved thinking
  * Claude 4.6+ and 5.x models: `thinking={"type": "adaptive"}`. Adaptive
    thinking interleaves between tool calls automatically, no beta header.
    These models reject `type: "enabled"` (4.7+) and default to
    display="omitted" (empty thinking text), so we ask for "summarized"
    to make reasoning visible in the CLI.
  * Claude 4.0-4.5 / 3.7: manual `{"type": "enabled", "budget_tokens": N}`
    plus the `interleaved-thinking-2025-05-14` beta header. max_tokens
    must exceed budget_tokens (the original scaffold sent 4096/4096,
    which the API rejects).
  * Thinking blocks (and redacted_thinking blocks) are replayed verbatim
    with their signatures. ThinkingBlocks produced by *other* providers
    are never sent (an OpenAI reasoning id is not a Claude signature).

Prompt caching (explicit breakpoints, max 4 per request)
  1. system prompt -- prefix order is tools -> system -> messages, so this
     one breakpoint caches the tool definitions too;
  2. the last block of the final message -- the "moving" breakpoint that
     writes the whole conversation so far into the cache each call;
  3. the last block of the previous user turn -- where the *previous*
     request's moving breakpoint was, guaranteeing a cache read even if
     a long tool iteration added more than the 20-block lookback window;
  4. at most one user-pinned breakpoint from state.cache_breakpoints.
  Thinking blocks can't carry cache_control, so we walk back to the
  nearest eligible block. UsageEvent reports cache_read/cache_write.

Resume invariants after an interrupt (or a provider switch)
  * every tool_use must be answered by a tool_result in the *immediately
    following* user message (tool_result blocks first). The loop already
    records results on interrupt; `_repair_tool_pairs` is defence in depth
    and inserts synthetic results *in place* (the scaffold appended them
    at the very end, which is rejected when a newer user message exists);
  * roles must alternate, so consecutive same-role messages (e.g. tool
    results + interrupt note + next user message) are merged;
  * empty text blocks and empty messages are rejected, so they're dropped;
  * manual thinking mode requires the final assistant turn of a tool loop
    to start with a thinking block. If that turn came from another
    provider (switched mid-loop), thinking is disabled for that one
    request and a Notice is emitted.
"""
from __future__ import annotations

import copy
import re
from typing import Any, AsyncIterator, Literal

from ..core.state import ConversationState, Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from ..tools.base import ToolDefinition
from .base import (
    AgentProvider, MessageComplete, Notice, StreamEvent, TextDelta, ThinkingDelta,
    ToolCallArgsDelta, ToolCallEnd, ToolCallStart, UsageEvent, error_from_exception,
    parse_tool_arguments,
)

ThinkingMode = Literal["adaptive", "manual", "none"]
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
MAX_BREAKPOINTS = 4
_EPHEMERAL = {"type": "ephemeral"}

_MODEL_RE = re.compile(
    r"claude-(?:(?P<fam>opus|sonnet|haiku|fable|mythos)-(?P<maj>\d+)(?:-(?P<min>\d{1,2})(?!\d))?"
    r"|(?P<maj2>\d)-(?P<min2>\d)-(?P<fam2>opus|sonnet|haiku))"
)


def thinking_mode_for(model: str) -> ThinkingMode:
    """Which thinking API shape a model accepts. Unknown/future models get
    the newest shape (adaptive)."""
    m = _MODEL_RE.search(model.lower())
    if not m:
        return "adaptive"
    if m["fam"]:
        fam, ver = m["fam"], (int(m["maj"]), int(m["min"] or 0))
    else:
        fam, ver = m["fam2"], (int(m["maj2"]), int(m["min2"]))
    if fam in ("fable", "mythos") or ver >= (4, 6):
        return "adaptive"
    if ver >= (4, 0) or ver == (3, 7):
        return "manual"
    return "none"


class AnthropicProvider(AgentProvider):
    name = "anthropic"

    def __init__(
        self,
        client: Any,
        enable_thinking: bool = True,
        thinking_budget_tokens: int = 10_000,
        max_tokens: int = 32_000,
        thinking_display: Literal["summarized", "omitted"] = "summarized",
        effort: str | None = None,
        thinking_mode: ThinkingMode | None = None,  # override auto-detection
        system_prompt: str | None = None,           # override state.system_prompt
    ):
        self.client = client
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        self.max_tokens = max_tokens
        self.thinking_display = thinking_display
        self.effort = effort
        self.thinking_mode = thinking_mode
        self.system_prompt = system_prompt

    def supports_interleaved_thinking(self, model: str | None = None) -> bool:
        return self.enable_thinking and (model is None or self._mode(model) != "none")

    def supports_prompt_caching(self) -> bool:
        return True

    def build_cache_control(self, state: ConversationState) -> dict:
        return {"pinned": [b.after_message_index for b in state.cache_breakpoints], "auto": ["system", "tail", "previous_tail"]}

    def _mode(self, model: str) -> ThinkingMode:
        if not self.enable_thinking:
            return "none"
        return self.thinking_mode or thinking_mode_for(model)

    # ------------------------------------------------------------ wire format

    def _convert(self, msg: Message) -> tuple[str, list[dict]]:
        if msg.role == "user":
            return "user", [{"type": "text", "text": b.text} for b in msg.content
                            if isinstance(b, TextBlock) and b.text.strip()]
        if msg.role == "tool":
            return "user", [{
                "type": "tool_result",
                "tool_use_id": b.tool_use_id,
                "content": b.output or "(no output)",
                "is_error": b.is_error,
            } for b in msg.content if isinstance(b, ToolResultBlock)]
        blocks: list[dict] = []
        for b in msg.content:
            if isinstance(b, ThinkingBlock):
                if b.provider not in (None, self.name) or not b.signature:
                    continue  # foreign or unsigned reasoning: not replayable here
                if b.redacted:
                    blocks.append({"type": "redacted_thinking", "data": b.signature})
                else:
                    blocks.append({"type": "thinking", "thinking": b.text, "signature": b.signature})
            elif isinstance(b, TextBlock):
                if b.text.strip():
                    blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolUseBlock):
                blocks.append({"type": "tool_use", "id": b.id, "name": b.tool_name, "input": b.input})
        return "assistant", blocks

    def _to_anthropic_messages(self, state: ConversationState) -> list[dict]:
        wire, _ = self._to_wire(state)
        return wire

    def _to_wire(self, state: ConversationState) -> tuple[list[dict], dict[int, int]]:
        """Returns (messages, map from state.messages index -> wire index)."""
        wire: list[dict] = []
        index_map: dict[int, int] = {}
        for idx, msg in enumerate(state.messages):
            role, blocks = self._convert(msg)
            if blocks:
                if wire and wire[-1]["role"] == role:
                    wire[-1]["content"].extend(blocks)
                else:
                    wire.append({"role": role, "content": blocks})
            if wire:
                index_map[idx] = len(wire) - 1
        while wire and wire[0]["role"] != "user":
            wire.pop(0)
        _repair_tool_pairs(wire)
        return wire, index_map

    def _tools_schema(self, tools: list[ToolDefinition]) -> list[dict]:
        # No cache_control here: the system-prompt breakpoint already
        # covers tools (they precede it in the cache prefix), saving a slot.
        return [t.to_anthropic_schema() for t in tools]

    def _apply_cache_breakpoints(self, wire: list[dict], index_map: dict[int, int], state: ConversationState) -> None:
        targets: list[int] = []
        if wire:
            targets.append(len(wire) - 1)
        user_idx = [i for i, m in enumerate(wire) if m["role"] == "user"]
        if len(user_idx) >= 2:
            targets.append(user_idx[-2])
        budget = MAX_BREAKPOINTS - 1 - len(targets)  # 1 slot is the system prompt
        for bp in state.cache_breakpoints:
            if budget <= 0:
                break
            w = index_map.get(bp.after_message_index)
            if w is not None and w not in targets:
                targets.append(w)
                budget -= 1
        for w in targets:
            content = wire[w]["content"]
            for j in range(len(content) - 1, -1, -1):
                if content[j]["type"] in ("thinking", "redacted_thinking"):
                    continue
                content[j] = {**content[j], "cache_control": _EPHEMERAL}
                break

    def build_request(self, state: ConversationState, tools: list[ToolDefinition], model: str) -> tuple[dict, list[str]]:
        notices: list[str] = []
        wire, index_map = self._to_wire(state)
        wire = copy.deepcopy(wire)
        mode = self._mode(model)
        if mode == "manual" and _tail_is_tool_loop_without_thinking(wire):
            mode = "none"
            notices.append(
                "Thinking disabled for this request: the in-progress tool loop was started by "
                "another provider/model, and manual-thinking Claude models require the last "
                "assistant turn of a tool loop to begin with their own thinking block."
            )
        self._apply_cache_breakpoints(wire, index_map, state)

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": self.system_prompt or state.system_prompt, "cache_control": _EPHEMERAL}],
            messages=wire,
        )
        if tools:
            kwargs["tools"] = self._tools_schema(tools)
        if mode == "adaptive":
            kwargs["thinking"] = {"type": "adaptive", "display": self.thinking_display}
            if self.effort:
                kwargs["output_config"] = {"effort": self.effort}
        elif mode == "manual":
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}
            kwargs["max_tokens"] = max(self.max_tokens, self.thinking_budget_tokens + 4096)
            kwargs["extra_headers"] = {"anthropic-beta": INTERLEAVED_THINKING_BETA}
        return kwargs, notices

    # -------------------------------------------------------------- streaming

    async def stream_turn(
        self, state: ConversationState, tools: list[ToolDefinition], model: str
    ) -> AsyncIterator[StreamEvent]:
        kwargs, notices = self.build_request(state, tools, model)
        for n in notices:
            yield Notice(text=n)

        tool_use_by_index: dict[int, str] = {}
        try:
            # `async with` guarantees the HTTP stream is closed even when
            # the consuming task is cancelled mid-stream (CancelledError is
            # a BaseException, so the `except Exception` below never eats it).
            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    et = event.type
                    if et == "content_block_start" and event.content_block.type == "tool_use":
                        block = event.content_block
                        tool_use_by_index[event.index] = block.id
                        yield ToolCallStart(id=block.id, tool_name=block.name)
                    elif et == "content_block_delta":
                        d = event.delta
                        if d.type == "text_delta":
                            yield TextDelta(text=d.text)
                        elif d.type == "thinking_delta":
                            yield ThinkingDelta(text=d.thinking)
                        elif d.type == "input_json_delta" and event.index in tool_use_by_index:
                            yield ToolCallArgsDelta(id=tool_use_by_index[event.index], partial_json=d.partial_json)
                    elif et == "content_block_stop" and event.index in tool_use_by_index:
                        yield ToolCallEnd(id=tool_use_by_index[event.index])
                final = await stream.get_final_message()
        except Exception as exc:
            yield error_from_exception(exc)
            return

        blocks: list = []
        for block in final.content:
            bt = block.type
            if bt == "text":
                blocks.append(TextBlock(text=block.text))
            elif bt == "thinking":
                blocks.append(ThinkingBlock(text=block.thinking, signature=getattr(block, "signature", None), provider=self.name))
            elif bt == "redacted_thinking":
                blocks.append(ThinkingBlock(redacted=True, signature=block.data, provider=self.name))
            elif bt == "tool_use":
                inp = block.input if isinstance(block.input, dict) else parse_tool_arguments(str(block.input))
                blocks.append(ToolUseBlock(id=block.id, tool_name=block.name, input=inp))

        stop = getattr(final, "stop_reason", None)
        if stop == "max_tokens" and any(isinstance(b, ToolUseBlock) for b in blocks):
            # A tool call cut off by max_tokens has truncated arguments;
            # executing it would be wrong. Drop it and tell the user.
            blocks = [b for b in blocks if not isinstance(b, ToolUseBlock)]
            yield Notice(text="Response hit max_tokens mid tool call; the truncated call was dropped.")
        elif stop == "refusal":
            yield Notice(text="Claude declined to continue this request (stop_reason=refusal).")

        usage = getattr(final, "usage", None)
        if usage is not None:
            yield UsageEvent(
                provider=self.name, model=model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            )
        yield MessageComplete(message=Message(role="assistant", content=blocks, provider=self.name, model=model))


def _repair_tool_pairs(wire: list[dict]) -> None:
    """Make every tool_use/tool_result pairing valid, in place."""
    i = 0
    while i < len(wire):
        msg = wire[i]
        if msg["role"] == "assistant":
            ids = [b["id"] for b in msg["content"] if b["type"] == "tool_use"]
            if ids:
                nxt = wire[i + 1] if i + 1 < len(wire) else None
                if nxt is None or nxt["role"] != "user":
                    nxt = {"role": "user", "content": []}
                    wire.insert(i + 1, nxt)
        else:
            prev = wire[i - 1] if i > 0 else None
            allowed = [b["id"] for b in prev["content"] if b["type"] == "tool_use"] if prev and prev["role"] == "assistant" else []
            results = {b["tool_use_id"]: b for b in msg["content"] if b["type"] == "tool_result" and b["tool_use_id"] in allowed}
            others = [b for b in msg["content"] if b["type"] != "tool_result"]
            ordered = [
                results.get(tid) or {
                    "type": "tool_result", "tool_use_id": tid, "is_error": True,
                    "content": "Cancelled by user (interrupted before this tool call completed).",
                }
                for tid in allowed
            ]
            msg["content"] = ordered + others  # tool_result blocks must come first
            if not msg["content"]:
                wire.pop(i)
                continue
        i += 1


def _tail_is_tool_loop_without_thinking(wire: list[dict]) -> bool:
    if len(wire) < 2 or wire[-1]["role"] != "user":
        return False
    if not any(b["type"] == "tool_result" for b in wire[-1]["content"]):
        return False
    prev = wire[-2]
    return prev["role"] == "assistant" and prev["content"][0]["type"] not in ("thinking", "redacted_thinking")
