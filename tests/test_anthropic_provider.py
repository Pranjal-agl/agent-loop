"""
Exercises AnthropicProvider.stream_turn against a fake transport that
mimics the shape of anthropic.AsyncMessageStream, without any network
call. This is the same "catch structural bugs before they hit a real API
bill" pattern used for AgentLoop + MockProvider.
"""
from types import SimpleNamespace

import pytest

from agent_loop.core.state import ConversationState, Message, TextBlock, ToolUseBlock
from agent_loop.providers.anthropic_provider import AnthropicProvider
from agent_loop.providers.base import MessageComplete, TextDelta, ThinkingDelta, ToolCallStart


class FakeAnthropicStream:
    """Fakes the async-context-manager + async-iterator + get_final_message
    surface of anthropic.AsyncMessageStream."""

    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for e in self._events:
            yield e

    async def get_final_message(self):
        return self._final_message


class FakeMessagesResource:
    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeAnthropicStream(self._events, self._final_message)


class FakeClient:
    def __init__(self, events, final_message):
        self.messages = FakeMessagesResource(events, final_message)


def _text_events():
    return [
        SimpleNamespace(type="content_block_start", index=0,
                         content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", index=0,
                         delta=SimpleNamespace(type="text_delta", text="Hello ")),
        SimpleNamespace(type="content_block_delta", index=0,
                         delta=SimpleNamespace(type="text_delta", text="world")),
        SimpleNamespace(type="content_block_stop", index=0),
    ]


@pytest.mark.asyncio
async def test_text_streaming_produces_deltas_and_final_message():
    final = SimpleNamespace(content=[SimpleNamespace(type="text", text="Hello world")])
    client = FakeClient(_text_events(), final)
    provider = AnthropicProvider(client, enable_thinking=False)
    state = ConversationState(active_provider="anthropic", active_model="claude-x")
    state.messages.append(Message(role="user", content=[TextBlock(text="hi")]))

    events = [e async for e in provider.stream_turn(state, tools=[], model="claude-x")]

    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(d.text for d in deltas) == "Hello world"
    complete = [e for e in events if isinstance(e, MessageComplete)]
    assert len(complete) == 1
    assert complete[0].message.content[0].text == "Hello world"


@pytest.mark.asyncio
async def test_tool_use_streaming_produces_tool_call_start():
    events = [
        SimpleNamespace(type="content_block_start", index=0,
                         content_block=SimpleNamespace(type="tool_use", id="tu_1", name="bash")),
        SimpleNamespace(type="content_block_delta", index=0,
                         delta=SimpleNamespace(type="input_json_delta", partial_json='{"command"')),
        SimpleNamespace(type="content_block_delta", index=0,
                         delta=SimpleNamespace(type="input_json_delta", partial_json=': "ls"}')),
        SimpleNamespace(type="content_block_stop", index=0),
    ]
    final = SimpleNamespace(content=[
        SimpleNamespace(type="tool_use", id="tu_1", name="bash", input={"command": "ls"})
    ])
    client = FakeClient(events, final)
    provider = AnthropicProvider(client, enable_thinking=False)
    state = ConversationState(active_provider="anthropic", active_model="claude-x")
    state.messages.append(Message(role="user", content=[TextBlock(text="list files")]))

    seen = [e async for e in provider.stream_turn(state, tools=[], model="claude-x")]

    starts = [e for e in seen if isinstance(e, ToolCallStart)]
    assert starts and starts[0].tool_name == "bash"
    complete = next(e for e in seen if isinstance(e, MessageComplete))
    tool_block = complete.message.content[0]
    assert isinstance(tool_block, ToolUseBlock)
    assert tool_block.input == {"command": "ls"}


def test_dangling_tool_use_gets_synthetic_result_before_next_call():
    """The resume-invariant fix: an interrupted turn that emitted a
    tool_use with no tool_result must not produce an API-rejecting
    request on the next call."""
    provider = AnthropicProvider(client=None, enable_thinking=False)
    state = ConversationState(active_provider="anthropic", active_model="claude-x")
    state.messages.append(Message(role="user", content=[TextBlock(text="hi")]))
    state.messages.append(Message(
        role="assistant",
        content=[ToolUseBlock(id="tu_1", tool_name="bash", input={"command": "ls"})],
        partial=True,
    ))
    # no matching tool message appended -- simulates an interrupt

    messages = provider._to_anthropic_messages(state)
    last = messages[-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "tu_1"
    assert last["content"][0]["is_error"] is True


# ---------------------------------------------------------------------------
# Thinking configuration, caching, resume invariants
# ---------------------------------------------------------------------------
from agent_loop.core.state import ThinkingBlock, ToolResultBlock  # noqa: E402
from agent_loop.providers.anthropic_provider import (  # noqa: E402
    INTERLEAVED_THINKING_BETA, thinking_mode_for,
)
from agent_loop.providers.base import Notice, ProviderError, UsageEvent, error_from_exception  # noqa: E402
from agent_loop.tools.bash import BashTool  # noqa: E402
from wire_validators import assert_valid_anthropic  # noqa: E402


@pytest.mark.parametrize("model,mode", [
    ("claude-opus-5-5", "adaptive"), ("claude-sonnet-5-5", "adaptive"), ("claude-haiku-5-5", "adaptive"),
    ("claude-fable-5-2", "adaptive"), ("claude-opus-4-6", "adaptive"), ("claude-opus-4-7", "adaptive"),
    ("claude-sonnet-4-5", "manual"), ("claude-haiku-4-5", "manual"), ("claude-opus-4-5-20251101", "manual"),
    ("claude-sonnet-4-20250514", "manual"), ("claude-3-7-sonnet-latest", "manual"),
    ("claude-3-5-haiku-latest", "none"), ("some-future-model", "adaptive"),
])
def test_thinking_mode_detection(model, mode):
    assert thinking_mode_for(model) == mode


def _state(*msgs):
    s = ConversationState(active_provider="anthropic", active_model="claude-x")
    s.messages.extend(msgs)
    return s


def test_adaptive_request_shape(tmp_path):
    p = AnthropicProvider(client=None, effort="high")
    kwargs, notices = p.build_request(_state(Message(role="user", content=[TextBlock(text="hi")])),
                                      [BashTool(tmp_path)], "claude-opus-5-5")
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert "extra_headers" not in kwargs and not notices
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in kwargs["tools"][-1]  # system breakpoint covers tools


def test_manual_request_shape_has_valid_budget_and_beta_header():
    p = AnthropicProvider(client=None, max_tokens=4096, thinking_budget_tokens=4096)
    kwargs, _ = p.build_request(_state(Message(role="user", content=[TextBlock(text="hi")])), [], "claude-haiku-4-5")
    assert kwargs["thinking"]["type"] == "enabled"
    assert kwargs["max_tokens"] > kwargs["thinking"]["budget_tokens"]  # the scaffold sent 4096/4096
    assert kwargs["extra_headers"]["anthropic-beta"] == INTERLEAVED_THINKING_BETA


def test_dangling_tool_use_in_the_middle_is_repaired_in_place():
    """Interrupted tool call followed by a NEWER user message: the synthetic
    tool_result must go right after the tool_use (merged with the user's
    text), not appended at the end of the conversation."""
    p = AnthropicProvider(client=None)
    state = _state(
        Message(role="user", content=[TextBlock(text="run it")]),
        Message(role="assistant", content=[ThinkingBlock(text="t", signature="sig", provider="anthropic"),
                                           ToolUseBlock(id="tu_1", tool_name="bash", input={"command": "ls"})]),
        Message(role="user", content=[TextBlock(text="[interrupted]")], synthetic=True),
        Message(role="user", content=[TextBlock(text="do something else")]),
    )
    msgs = p._to_anthropic_messages(state)
    assert_valid_anthropic(msgs)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result", "text", "text"]


def test_foreign_and_unsigned_thinking_are_dropped_redacted_kept():
    p = AnthropicProvider(client=None)
    state = _state(
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", provider="openai", content=[
            ThinkingBlock(text="openai summary", provider="openai", provider_meta={"id": "rs_1", "encrypted_content": "x"}),
            TextBlock(text="from gpt"),
        ]),
        Message(role="user", content=[TextBlock(text="q2")]),
        Message(role="assistant", content=[
            ThinkingBlock(text="mine", signature="SIG", provider="anthropic"),
            ThinkingBlock(redacted=True, signature="ENCRYPTED", provider="anthropic"),
            ThinkingBlock(text="partial, unsigned", provider="anthropic"),
            TextBlock(text="from claude"),
        ]),
        Message(role="user", content=[TextBlock(text="q3")]),
    )
    msgs = p._to_anthropic_messages(state)
    assert_valid_anthropic(msgs)
    assert [b["type"] for b in msgs[1]["content"]] == ["text"]
    assert [b["type"] for b in msgs[3]["content"]] == ["thinking", "redacted_thinking", "text"]
    assert msgs[3]["content"][1]["data"] == "ENCRYPTED"


def test_cache_breakpoints_move_with_the_conversation():
    p = AnthropicProvider(client=None)
    state = _state(
        Message(role="user", content=[TextBlock(text="q1")]),
        Message(role="assistant", content=[ThinkingBlock(text="t", signature="s", provider="anthropic"),
                                           ToolUseBlock(id="a", tool_name="bash", input={})]),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="a", output="out1")]),
        Message(role="assistant", content=[ToolUseBlock(id="b", tool_name="bash", input={}),
                                           ThinkingBlock(text="t2", signature="s2", provider="anthropic")]),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="b", output="out2")]),
    )
    kwargs, _ = p.build_request(state, [], "claude-opus-5-5")
    msgs = kwargs["messages"]
    assert_valid_anthropic(msgs)
    marked = [(i, b["type"]) for i, m in enumerate(msgs) for b in m["content"] if "cache_control" in b]
    # tail (latest tool results) + previous tail (previous tool results)
    assert marked == [(2, "tool_result"), (4, "tool_result")]
    # state itself is never mutated by request building
    kwargs2, _ = p.build_request(state, [], "claude-opus-5-5")
    assert kwargs2["messages"] == msgs


def test_cache_breakpoint_skips_thinking_blocks():
    p = AnthropicProvider(client=None)
    state = _state(Message(role="user", content=[TextBlock(text="q")]))
    kwargs, _ = p.build_request(state, [], "claude-opus-5-5")
    assert kwargs["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_manual_mode_disables_thinking_for_foreign_tool_loop():
    p = AnthropicProvider(client=None)
    state = _state(
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", provider="openai", content=[ToolUseBlock(id="call_1", tool_name="bash", input={})]),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="call_1", output="x")]),
    )
    kwargs, notices = p.build_request(state, [], "claude-sonnet-4-5")
    assert "thinking" not in kwargs and notices
    kwargs, notices = p.build_request(state, [], "claude-opus-5-5")  # adaptive relaxes the rule
    assert kwargs["thinking"]["type"] == "adaptive" and not notices


@pytest.mark.asyncio
async def test_thinking_signature_and_cache_usage_are_captured():
    events = [
        SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="thinking")),
        SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="thinking_delta", thinking="hmm")),
        SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="signature_delta", signature="SIG")),
        SimpleNamespace(type="content_block_stop", index=0),
    ]
    final = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="hmm", signature="SIG"),
                 SimpleNamespace(type="redacted_thinking", data="ENC"),
                 SimpleNamespace(type="text", text="answer")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=900, cache_creation_input_tokens=50),
    )
    provider = AnthropicProvider(FakeClient(events, final))
    seen = [e async for e in provider.stream_turn(_state(Message(role="user", content=[TextBlock(text="hi")])), [], "claude-opus-5-5")]
    assert any(isinstance(e, ThinkingDelta) for e in seen)
    usage = next(e for e in seen if isinstance(e, UsageEvent))
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (900, 50)
    blocks = seen[-1].message.content
    assert blocks[0].signature == "SIG" and blocks[0].provider == "anthropic"
    assert blocks[1].redacted and blocks[1].signature == "ENC"


@pytest.mark.asyncio
async def test_tool_call_truncated_by_max_tokens_is_dropped():
    final = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="let me"),
                 SimpleNamespace(type="tool_use", id="t", name="bash", input={"comm": ""})],
        stop_reason="max_tokens", usage=None,
    )
    provider = AnthropicProvider(FakeClient([], final), enable_thinking=False)
    seen = [e async for e in provider.stream_turn(_state(Message(role="user", content=[TextBlock(text="hi")])), [], "c")]
    assert any(isinstance(e, Notice) for e in seen)
    assert not any(isinstance(b, ToolUseBlock) for b in seen[-1].message.content)


@pytest.mark.asyncio
async def test_sdk_errors_become_provider_errors():
    class Boom:
        def stream(self, **kw):
            exc = Exception("Overloaded")
            exc.status_code = 529
            raise exc

    provider = AnthropicProvider(SimpleNamespace(messages=Boom()), enable_thinking=False)
    seen = [e async for e in provider.stream_turn(_state(Message(role="user", content=[TextBlock(text="hi")])), [], "c")]
    assert isinstance(seen[-1], ProviderError) and seen[-1].recoverable


def test_error_classification():
    class E(Exception):
        pass

    def make(status):
        e = E("x")
        e.status_code = status
        return e

    assert error_from_exception(make(429)).recoverable
    assert error_from_exception(make(503)).recoverable
    assert not error_from_exception(make(400)).recoverable
    assert error_from_exception(type("APIConnectionError", (Exception,), {})("net")).recoverable
