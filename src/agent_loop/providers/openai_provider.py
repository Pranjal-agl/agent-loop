"""
OpenAI adapter, built on the Responses API rather than Chat Completions.
Rationale (this is the "optimized for reasoning models" requirement from
the spec): Responses API is what surfaces reasoning items/summaries for
o-series and gpt-5.x reasoning models and supports passing prior
reasoning items back on the next turn, which Chat Completions does not
expose. Field/event names below verified against openai==3.10.0's actual
type definitions via introspection in this environment -- NOT tested
against a live API call, since no OPENAI_API_KEY is available here and
api.openai.com is outside this sandbox's network allowlist regardless.
See tests/test_openai_provider.py for a mocked-transport test.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from openai import AsyncOpenAI

from ..core.state import ConversationState, Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from ..tools.base import ToolDefinition
from .base import (
    AgentProvider, MessageComplete, ProviderError, StreamEvent, TextDelta, ThinkingDelta,
    ToolCallArgsDelta, ToolCallEnd, ToolCallStart,
)


class OpenAIProvider(AgentProvider):
    """Adapter for OpenAI models via the Responses API.

    Provider-specific behavior / asymmetries with Claude, documented here
    rather than left implicit:

    - No explicit prompt-cache breakpoints. OpenAI caches automatically
      based on a stable prompt prefix (`prompt_cache_key` groups requests
      onto the same cache) -- there's no per-block `cache_control` to set.
      `build_cache_control` is therefore a documented no-op for this
      provider; state.cache_breakpoints is simply ignored here.
    - "Interleaved thinking" has no OpenAI equivalent to hand-roll: the
      model's reasoning is either not exposed at all, or exposed only as
      a reasoning *summary* (`reasoning.summary`), never the raw chain of
      thought. `supports_interleaved_thinking()` is False for that reason
      -- ThinkingDelta events below carry the *summary* text when present,
      which is a strictly weaker signal than Claude's thinking blocks.
    - Resume invariant: unlike Claude, a dangling function_call with no
      function_call_output is NOT rejected by the API -- it can simply be
      omitted from the next request's `input`. `_to_input_items` therefore
      drops any assistant function_call that never got a matching
      ToolResultBlock, instead of needing to synthesize one.
    """

    name = "openai"

    def __init__(self, client: AsyncOpenAI, reasoning_effort: str | None = "medium"):
        self.client = client
        # None disables the `reasoning` param entirely (needed for
        # non-reasoning models, which reject the field outright).
        self.reasoning_effort = reasoning_effort

    def supports_interleaved_thinking(self) -> bool:
        return False

    def supports_prompt_caching(self) -> bool:
        return False

    def build_cache_control(self, state: ConversationState) -> dict:
        return {}

    # -- wire format conversion ------------------------------------------

    def _to_input_items(self, state: ConversationState) -> list[dict]:
        items: list[dict] = []
        # call_id -> index of its function_call item in `items`, so we can
        # retroactively drop it if no matching output ever arrives.
        pending_call_index: dict[str, int] = {}

        for msg in state.messages:
            if msg.role == "user":
                text = " ".join(b.text for b in msg.content if isinstance(b, TextBlock))
                items.append({"type": "message", "role": "user", "content": text})

            elif msg.role == "assistant":
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        items.append({"type": "message", "role": "assistant", "content": b.text})
                    elif isinstance(b, ThinkingBlock):
                        # Encrypted/opaque reasoning content, when the API
                        # returned one, should be round-tripped verbatim
                        # via b.signature (used here as the opaque blob
                        # field) rather than re-derived -- reasoning
                        # models require their own prior reasoning back
                        # unmodified for correct multi-turn behavior.
                        if b.signature:
                            items.append({"type": "reasoning", "id": b.signature, "summary": []})
                    elif isinstance(b, ToolUseBlock):
                        pending_call_index[b.id] = len(items)
                        items.append({
                            "type": "function_call",
                            "call_id": b.id,
                            "name": b.tool_name,
                            "arguments": json.dumps(b.input),
                        })

            elif msg.role == "tool":
                for b in msg.content:
                    if isinstance(b, ToolResultBlock):
                        items.append({
                            "type": "function_call_output",
                            "call_id": b.tool_use_id,
                            "output": b.output,
                        })
                        pending_call_index.pop(b.tool_use_id, None)

        # Drop any function_call left with no output -- see class
        # docstring: OpenAI doesn't require a synthetic result the way
        # Claude does.
        for idx in sorted(pending_call_index.values(), reverse=True):
            del items[idx]

        return items

    def _tools_schema(self, tools: list[ToolDefinition]) -> list[dict]:
        return [t.to_openai_schema()["function"] | {"type": "function"} for t in tools]

    # -- streaming ---------------------------------------------------------

    async def stream_turn(
        self, state: ConversationState, tools: list[ToolDefinition], model: str
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = dict(model=model, input=self._to_input_items(state))
        if tools:
            kwargs["tools"] = self._tools_schema(tools)
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}

        # call_id -> synthetic StreamEvent id (Responses API function
        # calls are identified by call_id, item_id varies by event type,
        # so we track both to be safe).
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
                    elif et == "response.function_call_arguments.delta":
                        call_id = call_id_by_item_id.get(event.item_id, event.item_id)
                        yield ToolCallArgsDelta(id=call_id, partial_json=event.delta)
                    elif et == "response.output_item.done" and event.item.type == "function_call":
                        yield ToolCallEnd(id=event.item.call_id)

                final = await stream.get_final_response()
        except Exception as exc:
            # asyncio.CancelledError is a BaseException and is not caught
            # here -- see AnthropicProvider for the same note. This keeps
            # interrupt propagation identical across both adapters.
            yield ProviderError(message=str(exc), recoverable=False)
            return

        content_blocks = []
        for item in final.output:
            if item.type == "message":
                for c in item.content:
                    if getattr(c, "type", None) == "output_text":
                        content_blocks.append(TextBlock(text=c.text))
            elif item.type == "reasoning":
                summary_text = " ".join(s.text for s in getattr(item, "summary", []) if hasattr(s, "text"))
                # Round-trip the opaque item id via `signature` so a later
                # turn can send this exact reasoning item back -- see
                # _to_input_items.
                content_blocks.append(ThinkingBlock(text=summary_text, signature=item.id))
            elif item.type == "function_call":
                content_blocks.append(
                    ToolUseBlock(id=item.call_id, tool_name=item.name, input=json.loads(item.arguments or "{}"))
                )

        yield MessageComplete(message=Message(role="assistant", content=content_blocks))
