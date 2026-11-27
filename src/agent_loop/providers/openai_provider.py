"""
OpenAI adapter, built on the Responses API rather than Chat Completions.

Why Responses: it is the API that exposes reasoning items for reasoning
models (gpt-5.x, codex, o-series) and lets you pass them back on the
next call, so the model keeps its chain of thought across tool calls
(the OpenAI analogue of Claude's interleaved thinking). Chat Completions
throws reasoning away between calls.

Provider-specific behaviour
---------------------------
* Stateless by design: `store=False` + `include=["reasoning.encrypted_content"]`.
  The conversation lives in *our* ConversationState, not on OpenAI's
  servers (no previous_response_id). That is what makes switching
  Claude <-> OpenAI mid-conversation possible at all, and it also works
  for zero-data-retention orgs.
* Reasoning replay: each reasoning item is stored as a ThinkingBlock with
  provider="openai" and provider_meta={"id", "encrypted_content"} and
  sent back verbatim. Claude's thinking blocks are dropped (unreadable).
  Invariant: a reasoning item must be followed by the item it produced,
  so if a message's function calls are dropped, its reasoning goes too.
* Prompt caching is automatic on OpenAI (prefix-based, no explicit
  breakpoints). We set `prompt_cache_key` to the conversation id so all
  calls of one conversation route to the same cache, and report
  `cached_tokens` in UsageEvent.
* Resume invariants: every function_call needs a function_call_output
  somewhere later in `input` ("No tool output found" otherwise), and every
  output needs its call. The loop records results on interrupt, so this
  is defence in depth: unanswered calls are dropped, orphan outputs too.
  Unlike Claude there is no "immediately next message" or role-
  alternation rule, so consecutive user messages need no merging.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from ..core.state import ConversationState, Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from ..tools.base import ToolDefinition
from .base import (
    AgentProvider, MessageComplete, Notice, ProviderError, StreamEvent, TextDelta, ThinkingDelta,
    ToolCallArgsDelta, ToolCallEnd, ToolCallStart, UsageEvent, error_from_exception,
    parse_tool_arguments,
)

_REASONING_RE = re.compile(r"^(o\d|gpt-5|gpt-6|codex)")


def is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return ("codex" in m or bool(_REASONING_RE.match(m))) and "chat" not in m


class OpenAIProvider(AgentProvider):
    name = "openai"

    def __init__(self, client: Any, reasoning_effort: str | None = "medium", reasoning: bool | None = None):
        self.client = client
        self.reasoning_effort = reasoning_effort
        # None = auto-detect from the model name; reasoning params are
        # rejected by non-reasoning models (e.g. gpt-4.1).
        self.reasoning = reasoning

    def _reasoning_on(self, model: str) -> bool:
        if self.reasoning_effort is None:
            return False
        return self.reasoning if self.reasoning is not None else is_reasoning_model(model)

    def supports_interleaved_thinking(self, model: str | None = None) -> bool:
        # Reasoning persists across tool calls when encrypted items are
        # replayed, but only a *summary* is ever visible.
        return model is not None and self._reasoning_on(model)

    def supports_prompt_caching(self) -> bool:
        return True  # automatic, prefix-based

    # ------------------------------------------------------------ wire format

    def _to_input_items(self, state: ConversationState) -> list[dict]:
        answered = {b.tool_use_id for m in state.messages if m.role == "tool"
                    for b in m.content if isinstance(b, ToolResultBlock)}
        emitted_calls: set[str] = set()
        items: list[dict] = []

        for msg in state.messages:
            if msg.role == "user":
                text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock) and b.text.strip())
                if text:
                    items.append({"type": "message", "role": "user", "content": text})

            elif msg.role == "assistant":
                calls = [b for b in msg.content if isinstance(b, ToolUseBlock)]
                dropped_any = any(c.id not in answered for c in calls)
                group: list[dict] = []
                for b in msg.content:
                    if isinstance(b, ThinkingBlock):
                        meta = b.provider_meta
                        if b.provider == self.name and meta.get("encrypted_content") and not dropped_any:
                            group.append({
                                "type": "reasoning",
                                "id": meta["id"],
                                "summary": [{"type": "summary_text", "text": b.text}] if b.text else [],
                                "encrypted_content": meta["encrypted_content"],
                            })
                    elif isinstance(b, TextBlock):
                        if b.text.strip():
                            group.append({"type": "message", "role": "assistant", "content": b.text})
                    elif isinstance(b, ToolUseBlock):
                        if b.id not in answered:
                            continue
                        raw = b.input.get("__raw_arguments__")
                        group.append({
                            "type": "function_call",
                            "call_id": b.id,
                            "name": b.tool_name,
                            "arguments": raw if isinstance(raw, str) else json.dumps(b.input),
                        })
                        emitted_calls.add(b.id)
                # a reasoning item must be followed by the output it produced
                while group and group[-1]["type"] == "reasoning":
                    group.pop()
                items.extend(group)

            elif msg.role == "tool":
                for b in msg.content:
                    if isinstance(b, ToolResultBlock) and b.tool_use_id in emitted_calls:
                        items.append({
                            "type": "function_call_output",
                            "call_id": b.tool_use_id,
                            "output": b.output or "(no output)",
                        })
        return items

    def _tools_schema(self, tools: list[ToolDefinition]) -> list[dict]:
        return [t.to_openai_schema()["function"] | {"type": "function"} for t in tools]

    def build_request(self, state: ConversationState, tools: list[ToolDefinition], model: str) -> dict:
        kwargs: dict[str, Any] = dict(
            model=model,
            input=self._to_input_items(state),
            instructions=state.system_prompt,
            store=False,
            prompt_cache_key=f"agent-loop-{state.id}"[:64],
        )
        if tools:
            kwargs["tools"] = self._tools_schema(tools)
        if self._reasoning_on(model):
            kwargs["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
            kwargs["include"] = ["reasoning.encrypted_content"]
        return kwargs

    # -------------------------------------------------------------- streaming

    async def stream_turn(
        self, state: ConversationState, tools: list[ToolDefinition], model: str
    ) -> AsyncIterator[StreamEvent]:
        kwargs = self.build_request(state, tools, model)
        call_id_by_item_id: dict[str, str] = {}
        try:
            async with self.client.responses.stream(**kwargs) as stream:
                async for event in stream:
                    et = event.type
                    if et == "response.output_item.added" and event.item.type == "function_call":
                        call_id_by_item_id[event.item.id] = event.item.call_id
                        yield ToolCallStart(id=event.item.call_id, tool_name=event.item.name)
                    elif et == "response.output_text.delta":
                        yield TextDelta(text=event.delta)
                    elif et in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                        yield ThinkingDelta(text=event.delta)
                    elif et == "response.reasoning_summary_part.done":
                        yield ThinkingDelta(text="\n")
                    elif et == "response.function_call_arguments.delta":
                        call_id = call_id_by_item_id.get(event.item_id, event.item_id)
                        yield ToolCallArgsDelta(id=call_id, partial_json=event.delta)
                    elif et == "response.output_item.done" and event.item.type == "function_call":
                        yield ToolCallEnd(id=event.item.call_id)
                final = await stream.get_final_response()
        except Exception as exc:
            yield error_from_exception(exc)
            return

        status = getattr(final, "status", None)
        if status == "failed":
            err = getattr(final, "error", None)
            yield ProviderError(message=f"OpenAI response failed: {getattr(err, 'message', err)}", recoverable=False)
            return
        if status == "incomplete":
            reason = getattr(getattr(final, "incomplete_details", None), "reason", "unknown")
            yield Notice(text=f"OpenAI response incomplete ({reason}).")

        blocks: list = []
        for item in final.output:
            if item.type == "message":
                for c in item.content:
                    if getattr(c, "type", None) == "output_text":
                        blocks.append(TextBlock(text=c.text))
            elif item.type == "reasoning":
                summary = "\n".join(s.text for s in (getattr(item, "summary", None) or []) if hasattr(s, "text"))
                blocks.append(ThinkingBlock(
                    text=summary, provider=self.name,
                    provider_meta={"id": item.id, "encrypted_content": getattr(item, "encrypted_content", None)},
                ))
            elif item.type == "function_call":
                blocks.append(ToolUseBlock(
                    id=item.call_id, tool_name=item.name, input=parse_tool_arguments(item.arguments),
                ))

        usage = getattr(final, "usage", None)
        if usage is not None:
            in_details = getattr(usage, "input_tokens_details", None)
            out_details = getattr(usage, "output_tokens_details", None)
            yield UsageEvent(
                provider=self.name, model=model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(in_details, "cached_tokens", 0) or 0,
                reasoning_tokens=getattr(out_details, "reasoning_tokens", 0) or 0,
            )
        yield MessageComplete(message=Message(role="assistant", content=blocks, provider=self.name, model=model))
