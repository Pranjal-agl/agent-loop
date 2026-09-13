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
