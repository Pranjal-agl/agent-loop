"""
Gemini adapter (google-genai SDK, streaming generate_content).

This file is the proof that the AgentProvider abstraction holds: adding
a third provider touched nothing in the loop, the state model or the
other adapters.

Provider-specific behaviour / invariants
----------------------------------------
* Thought signatures. Gemini 3+ attaches an opaque `thought_signature` to
  the first functionCall part of each step (and sometimes to text parts).
  It must be sent back *in the same part* on the next request or the API
  returns 400. We store it base64-encoded in the block's
  provider_meta["gemini"] and re-attach it on conversion. Unlike Claude
  (a separate thinking block) the signature rides on the call itself.
* Foreign history. A functionCall that came from Claude/OpenAI has no
  signature; Gemini's documented escape hatch is the sentinel
  "skip_thought_signature_validator" on the first call of that step.
* Thought summaries (include_thoughts=True) are shown to the user but not
  replayed: the signatures carry the reasoning state, the summaries don't.
* Function responses go in a user-role Content immediately after the
  model Content that made the calls, one per call; consecutive same-role
  Contents are merged (merging *Contents* is fine, merging *parts* with
  signatures is not).
* Caching is implicit (automatic) on Gemini; cached tokens are reported.
"""
from __future__ import annotations

import base64
import contextlib
from typing import Any, AsyncIterator

from google.genai import types  # module-level: the first import is slow (~0.5s)

from ..core.state import ConversationState, Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from ..tools.base import ToolDefinition
from .base import (
    AgentProvider, MessageComplete, Notice, ProviderError, StreamEvent, TextDelta, ThinkingDelta,
    ToolCallEnd, ToolCallStart, UsageEvent, error_from_exception,
)

SKIP_SIGNATURE = "skip_thought_signature_validator"


def _b64(sig: bytes | None) -> str | None:
    return base64.urlsafe_b64encode(sig).decode("ascii") if sig else None


def _unb64(sig: str) -> bytes:
    return base64.urlsafe_b64decode(sig)


class GeminiProvider(AgentProvider):
    name = "gemini"

    def __init__(self, client: Any, include_thoughts: bool = True):
        self.client = client
        self.include_thoughts = include_thoughts

    def supports_interleaved_thinking(self, model: str | None = None) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        return True  # implicit

    # ------------------------------------------------------------ wire format

    def _sig(self, block) -> str | None:
        return (block.provider_meta.get(self.name) or {}).get("thought_signature")

    def _to_contents(self, state: ConversationState) -> list[dict]:
        names = {b.id: b.tool_name for m in state.messages for b in m.content if isinstance(b, ToolUseBlock)}
        contents: list[dict] = []

        def add(role: str, parts: list[dict]) -> None:
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for msg in state.messages:
            if msg.role == "user":
                add("user", [{"text": b.text} for b in msg.content if isinstance(b, TextBlock) and b.text.strip()])
            elif msg.role == "tool":
                add("user", [{
                    "function_response": {
                        "id": b.tool_use_id,
                        "name": b.tool_name or names.get(b.tool_use_id, "unknown"),
                        "response": {"error": b.output} if b.is_error else {"output": b.output},
                    }
                } for b in msg.content if isinstance(b, ToolResultBlock)])
            else:
                parts: list[dict] = []
                first_call = True
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        part: dict = {"text": b.text}
                        if (sig := self._sig(b)) and msg.provider == self.name:
                            part["thought_signature"] = _unb64(sig)
                        parts.append(part)
                    elif isinstance(b, ToolUseBlock):
                        part = {"function_call": {"id": b.id, "name": b.tool_name, "args": b.input}}
                        sig = self._sig(b)
                        if sig:
                            part["thought_signature"] = _unb64(sig)
                        elif first_call and msg.provider != self.name:
                            part["thought_signature"] = _unb64(SKIP_SIGNATURE)
                        first_call = False
                        parts.append(part)
                    # ThinkingBlocks: not replayed (see module docstring)
                add("model", parts)

        while contents and contents[0]["role"] != "user":
            contents.pop(0)
        _repair_function_pairs(contents)
        return contents

    def build_request(self, state: ConversationState, tools: list[ToolDefinition], model: str) -> dict:
        config = types.GenerateContentConfig(
            system_instruction=state.system_prompt,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(include_thoughts=self.include_thoughts),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in tools
            ])] if tools else None,
        )
        return {"model": model, "contents": self._to_contents(state), "config": config}

    # -------------------------------------------------------------- streaming

    async def stream_turn(
        self, state: ConversationState, tools: list[ToolDefinition], model: str
    ) -> AsyncIterator[StreamEvent]:
        request = self.build_request(state, tools, model)
        blocks: list = []
        usage = None
        finish = None
        try:
            stream = await self.client.aio.models.generate_content_stream(**request)
            closer = contextlib.aclosing(stream) if hasattr(stream, "aclose") else contextlib.nullcontext(stream)
            async with closer:
                async for chunk in stream:
                    usage = getattr(chunk, "usage_metadata", None) or usage
                    cands = getattr(chunk, "candidates", None) or []
                    if not cands:
                        continue
                    finish = getattr(cands[0], "finish_reason", None) or finish
                    content = getattr(cands[0], "content", None)
                    for part in (getattr(content, "parts", None) or []):
                        for ev in self._absorb(part, blocks):
                            yield ev
        except Exception as exc:
            yield error_from_exception(exc)
            return

        finish_name = getattr(finish, "name", str(finish) if finish else "")
        if finish_name == "MALFORMED_FUNCTION_CALL":
            yield ProviderError(message="Gemini produced a malformed function call", recoverable=True)
            return
        if finish_name in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT"):
            yield Notice(text=f"Gemini stopped generation (finish_reason={finish_name}).")

        if usage is not None:
            yield UsageEvent(
                provider=self.name, model=model,
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                cache_read_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                reasoning_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
            )
        yield MessageComplete(message=Message(role="assistant", content=blocks, provider=self.name, model=model))

    def _absorb(self, part: Any, blocks: list) -> list[StreamEvent]:
        """Fold one streamed Part into `blocks`, returning events to emit."""
        events: list[StreamEvent] = []
        sig = _b64(getattr(part, "thought_signature", None))
        meta = {self.name: {"thought_signature": sig}} if sig else {}
        fc = getattr(part, "function_call", None)
        text = getattr(part, "text", None)
        last = blocks[-1] if blocks else None

        if fc is not None:
            call = ToolUseBlock(tool_name=fc.name, input=dict(fc.args or {}), provider_meta=meta)
            if getattr(fc, "id", None):
                call.id = fc.id
            blocks.append(call)
            events += [ToolCallStart(id=call.id, tool_name=call.tool_name), ToolCallEnd(id=call.id)]
        elif getattr(part, "thought", False):
            if text:
                events.append(ThinkingDelta(text=text))
                if isinstance(last, ThinkingBlock):
                    last.text += text
                else:
                    blocks.append(ThinkingBlock(text=text, provider=self.name))
        elif text is not None or sig:
            if text:
                events.append(TextDelta(text=text))
            if isinstance(last, TextBlock) and not self._sig(last) and (text or sig):
                # Streamed text arrives in many chunks; the signature (if
                # any) arrives on the final one, often with empty text.
                last.text += text or ""
                if sig:
                    last.provider_meta = meta
            elif text:
                blocks.append(TextBlock(text=text, provider_meta=meta))
        return events


def _repair_function_pairs(contents: list[dict]) -> None:
    i = 0
    while i < len(contents):
        c = contents[i]
        if c["role"] == "model":
            ids = [p["function_call"]["id"] for p in c["parts"] if "function_call" in p]
            if ids and (i + 1 >= len(contents) or contents[i + 1]["role"] != "user"):
                contents.insert(i + 1, {"role": "user", "parts": []})
        else:
            prev = contents[i - 1] if i > 0 else None
            calls = {p["function_call"]["id"]: p["function_call"]["name"]
                     for p in (prev["parts"] if prev and prev["role"] == "model" else []) if "function_call" in p}
            have = {p["function_response"]["id"]: p for p in c["parts"]
                    if "function_response" in p and p["function_response"]["id"] in calls}
            responses = [have.get(cid) or {"function_response": {
                "id": cid, "name": name, "response": {"error": "Cancelled by user before this call completed."},
            }} for cid, name in calls.items()]
            c["parts"] = responses + [p for p in c["parts"] if "function_response" not in p]
            if not c["parts"]:
                contents.pop(i)
                continue
        i += 1
