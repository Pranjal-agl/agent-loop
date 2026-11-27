from types import SimpleNamespace

import pytest
from google.genai import types
from wire_validators import assert_valid_gemini

from agent_loop.core.state import ConversationState, Message, TextBlock, ToolResultBlock, ToolUseBlock
from agent_loop.providers.base import MessageComplete, TextDelta, ThinkingDelta, ToolCallStart, UsageEvent
from agent_loop.providers.gemini_provider import SKIP_SIGNATURE, GeminiProvider, _b64
from agent_loop.tools.bash import BashTool


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self.chunks:
            yield c


class FakeModels:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last = None

    async def generate_content_stream(self, **kw):
        self.last = kw
        return FakeStream(self.chunks)


def client(chunks):
    return SimpleNamespace(aio=SimpleNamespace(models=FakeModels(chunks)))


def chunk(*parts, usage=None, finish=None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)), finish_reason=finish)],
        usage_metadata=usage,
    )


def part(text=None, thought=False, sig=None, fc=None):
    return SimpleNamespace(text=text, thought=thought, thought_signature=sig, function_call=fc)


@pytest.mark.asyncio
async def test_streaming_captures_signatures_on_the_right_parts():
    chunks = [
        chunk(part(text="Planning the edit", thought=True)),
        chunk(part(text="I'll ")),
        chunk(part(text="check."), part(text="", sig=b"TEXTSIG")),
        chunk(part(fc=SimpleNamespace(id=None, name="bash", args={"command": "ls"}), sig=b"CALLSIG"),
              part(fc=SimpleNamespace(id=None, name="bash", args={"command": "pwd"})),
              usage=SimpleNamespace(prompt_token_count=50, candidates_token_count=9,
                                    cached_content_token_count=32, thoughts_token_count=7),
              finish=SimpleNamespace(name="STOP")),
    ]
    p = GeminiProvider(client(chunks))
    state = ConversationState()
    state.messages.append(Message(role="user", content=[TextBlock(text="hi")]))
    seen = [e async for e in p.stream_turn(state, [], "gemini-3.5-flash")]

    assert "".join(e.text for e in seen if isinstance(e, TextDelta)) == "I'll check."
    assert any(isinstance(e, ThinkingDelta) for e in seen)
    assert sum(isinstance(e, ToolCallStart) for e in seen) == 2
    usage = next(e for e in seen if isinstance(e, UsageEvent))
    assert usage.cache_read_tokens == 32 and usage.reasoning_tokens == 7
    msg = next(e for e in seen if isinstance(e, MessageComplete)).message
    thinking, text, call1, call2 = msg.content
    assert text.text == "I'll check." and text.provider_meta["gemini"]["thought_signature"] == _b64(b"TEXTSIG")
    assert call1.provider_meta["gemini"]["thought_signature"] == _b64(b"CALLSIG")
    assert call2.provider_meta == {}

    # Round trip: signatures go back in the same parts; thought summary is not replayed.
    msg.provider = "gemini"
    state.messages.append(msg)
    state.messages.append(Message(role="tool", content=[
        ToolResultBlock(tool_use_id=call1.id, tool_name="bash", output="a.txt"),
        ToolResultBlock(tool_use_id=call2.id, tool_name="bash", output="/w", is_error=True),
    ]))
    contents = p._to_contents(state)
    assert_valid_gemini(contents)
    model_parts = contents[1]["parts"]
    assert model_parts[0]["thought_signature"] == b"TEXTSIG"
    assert model_parts[1]["thought_signature"] == b"CALLSIG"
    assert "thought_signature" not in model_parts[2]
    assert contents[2]["parts"][1]["function_response"]["response"] == {"error": "/w"}


def test_foreign_tool_calls_get_skip_validator_and_dangling_calls_are_answered():
    p = GeminiProvider(client([]))
    state = ConversationState()
    state.messages += [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", provider="anthropic", content=[
            ToolUseBlock(id="toolu_1", tool_name="bash", input={"command": "ls"}),
            ToolUseBlock(id="toolu_2", tool_name="bash", input={"command": "pwd"})]),
        # interrupted: no tool message
        Message(role="user", content=[TextBlock(text="new question")]),
    ]
    contents = p._to_contents(state)
    assert_valid_gemini(contents)
    calls = contents[1]["parts"]
    assert types.Part(**calls[0]).model_dump(mode="json", exclude_none=True)["thought_signature"] == SKIP_SIGNATURE
    assert "thought_signature" not in calls[1]
    assert [list(x)[0] for x in contents[2]["parts"]] == ["function_response", "function_response", "text"]


def test_request_validates_against_sdk_types(tmp_path):
    p = GeminiProvider(client([]))
    state = ConversationState()
    state.messages += [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", provider="gemini", content=[
            ToolUseBlock(id="c1", tool_name="bash", input={"command": "ls"},
                         provider_meta={"gemini": {"thought_signature": _b64(b"S")}})]),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="c1", tool_name="bash", output="x")]),
    ]
    req = p.build_request(state, [BashTool(tmp_path)], "gemini-3.5-flash")
    for c in req["contents"]:
        types.Content.model_validate(c)
    assert req["config"].automatic_function_calling.disable is True
    assert req["config"].tools[0].function_declarations[0].name == "bash"
