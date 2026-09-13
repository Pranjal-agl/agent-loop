"""
Claude adapter. Verified against anthropic==1.4.0's actual type surface
(field names for streaming events, `messages.stream()` signature) via
introspection in this environment -- NOT tested against a live API call,
since no ANTHROPIC_API_KEY is available here. See tests/test_anthropic_provider.py
for a mocked-transport test that exercises this code path without network access.
"""
from __future__ import annotations

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from ..core.state import (
    ConversationState, Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock,
)
from ..tools.base import ToolDefinition
from .base import (
    AgentProvider, MessageComplete, ProviderError, StreamEvent, TextDelta, ThinkingDelta,
    ToolCallArgsDelta, ToolCallEnd, ToolCallStart,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding agent with access to filesystem tools."


class AnthropicProvider(AgentProvider):
    """Adapter for Claude models via the Messages streaming API.

    Provider-specific behavior implemented here:

    - Extended thinking: enabled via `thinking={"type": "enabled",
      "budget_tokens": ...}`. Thinking blocks stream in as their own
      content-block type and are preserved in history as ThinkingBlock so
      a later turn can still see the model's own prior reasoning.

    - Explicit prompt caching: `cache_control: {"type": "ephemeral"}` is
      attached to (a) the system prompt, (b) the last tool definition
      (covers the whole static tool list in one breakpoint), and (c) the
      content block at each index in `state.cache_breakpoints` -- the
      parts of the payload that repeat unchanged turn to turn.

    - Resume invariant: Claude rejects a request if an assistant turn's
      `tool_use` block has no matching `tool_result` in the following
      user turn. If a turn is interrupted after a tool_use was emitted
      but before its result was recorded, `_to_anthropic_messages`
      synthesizes a "cancelled by user" tool_result so the conversation
      is always structurally valid on the next call, instead of the API
      call failing outright. OpenAI's Chat/Responses APIs are more
      permissive here (a dangling tool call is typically just dropped),
      which is the concrete provider asymmetry the spec asks about.
    """

    name = "anthropic"

    def __init__(
        self,
        client: AsyncAnthropic,
        enable_thinking: bool = True,
        thinking_budget_tokens: int = 4096,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def supports_interleaved_thinking(self) -> bool:
        return self.enable_thinking

    def supports_prompt_caching(self) -> bool:
        return True

    def build_cache_control(self, state: ConversationState) -> dict:
        # The real work happens inline in _to_anthropic_messages, since
        # cache_control attaches to specific content blocks rather than
        # being a single top-level request param. Exposed here mainly so
        # tests/callers can introspect which breakpoints are configured.
        return {"breakpoints": [b.after_message_index for b in state.cache_breakpoints]}

    # -- wire format conversion ------------------------------------------

    def _to_anthropic_messages(self, state: ConversationState) -> list[dict]:
        messages: list[dict] = []
        cache_after = {b.after_message_index for b in state.cache_breakpoints}
        open_tool_use_ids: list[str] = []

        for idx, msg in enumerate(state.messages):
            if msg.role == "user":
                content = [
                    {"type": "text", "text": b.text}
                    for b in msg.content if isinstance(b, TextBlock)
                ]
                messages.append({"role": "user", "content": content})

            elif msg.role == "assistant":
                content = []
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        content.append({"type": "text", "text": b.text})
                    elif isinstance(b, ThinkingBlock):
                        block: dict = {"type": "thinking", "thinking": b.text}
                        if b.signature:
                            block["signature"] = b.signature
                        content.append(block)
                    elif isinstance(b, ToolUseBlock):
                        content.append(
                            {"type": "tool_use", "id": b.id, "name": b.tool_name, "input": b.input}
                        )
                        open_tool_use_ids.append(b.id)
                messages.append({"role": "assistant", "content": content})

            elif msg.role == "tool":
                content = []
                for b in msg.content:
                    if isinstance(b, ToolResultBlock):
                        content.append({
                            "type": "tool_result",
                            "tool_use_id": b.tool_use_id,
                            "content": b.output,
                            "is_error": b.is_error,
                        })
                        if b.tool_use_id in open_tool_use_ids:
                            open_tool_use_ids.remove(b.tool_use_id)
                messages.append({"role": "user", "content": content})

            if idx in cache_after and messages and messages[-1]["content"]:
                messages[-1]["content"][-1] = {
                    **messages[-1]["content"][-1],
                    "cache_control": {"type": "ephemeral"},
                }

        if open_tool_use_ids:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "Cancelled by user (interrupted before this tool call completed).",
                        "is_error": True,
                    }
                    for tid in open_tool_use_ids
                ],
            })

        return messages

    def _tools_schema(self, tools: list[ToolDefinition]) -> list[dict]:
        schema = [t.to_anthropic_schema() for t in tools]
        if schema:
            schema[-1] = {**schema[-1], "cache_control": {"type": "ephemeral"}}
        return schema

    # -- streaming ---------------------------------------------------------

    async def stream_turn(
        self, state: ConversationState, tools: list[ToolDefinition], model: str
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = dict(
            model=model,
            max_tokens=self.max_tokens,
            system=[
                {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
            ],
            messages=self._to_anthropic_messages(state),
        )
        if tools:
            kwargs["tools"] = self._tools_schema(tools)
        if self.enable_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}

        tool_use_by_index: dict[int, dict] = {}

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                # `async with` guarantees __aexit__ (which closes the HTTP
                # stream) runs even if this coroutine is cancelled mid-loop
                # -- that's what makes mid-stream interruption clean here.
                async for event in stream:
                    et = event.type
                    if et == "content_block_start" and event.content_block.type == "tool_use":
                        block = event.content_block
                        tool_use_by_index[event.index] = {"id": block.id, "name": block.name}
                        yield ToolCallStart(id=block.id, tool_name=block.name)
                    elif et == "content_block_delta":
                        d = event.delta
                        if d.type == "text_delta":
                            yield TextDelta(text=d.text)
                        elif d.type == "thinking_delta":
                            yield ThinkingDelta(text=d.thinking)
                        elif d.type == "input_json_delta":
                            entry = tool_use_by_index.get(event.index)
                            if entry is not None:
                                yield ToolCallArgsDelta(id=entry["id"], partial_json=d.partial_json)
                    elif et == "content_block_stop":
                        entry = tool_use_by_index.get(event.index)
                        if entry is not None:
                            yield ToolCallEnd(id=entry["id"])

                final = await stream.get_final_message()
        except Exception as exc:
            # NOTE: asyncio.CancelledError is a BaseException in Python
            # 3.8+, so it is NOT caught by this `except Exception` and
            # propagates normally -- that's required for interrupt
            # handling in AgentLoop to see the cancellation.
            yield ProviderError(message=str(exc), recoverable=False)
            return

        content_blocks = []
        for block in final.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "thinking":
                content_blocks.append(
                    ThinkingBlock(text=block.thinking, signature=getattr(block, "signature", None))
                )
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(id=block.id, tool_name=block.name, input=block.input)
                )

        yield MessageComplete(message=Message(role="assistant", content=content_blocks))
