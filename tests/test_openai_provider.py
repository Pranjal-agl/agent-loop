"""
Exercises OpenAIProvider.stream_turn against a fake transport that mimics
the shape of the Responses API's streaming context manager, no network
call involved.
"""
from types import SimpleNamespace

import pytest

from agent_loop.core.state import ConversationState, Message, TextBlock, ToolResultBlock, ToolUseBlock
from agent_loop.providers.base import MessageComplete, TextDelta, ToolCallStart
from agent_loop.providers.openai_provider import OpenAIProvider


class FakeResponsesStream:
    def __init__(self, events, final_response):
        self._events = events
        self._final_response = final_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for e in self._events:
            yield e

    async def get_final_response(self):
        return self._final_response


class FakeResponsesResource:
    def __init__(self, events, final_response):
        self._events = events
        self._final_response = final_response
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponsesStream(self._events, self._final_response)


class FakeClient:
    def __init__(self, events, final_response):
        self.responses = FakeResponsesResource(events, final_response)


@pytest.mark.asyncio
async def test_text_streaming_produces_deltas_and_final_message():
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello "),
        SimpleNamespace(type="response.output_text.delta", delta="world"),
    ]
    final = SimpleNamespace(output=[
        SimpleNamespace(type="message", content=[
            SimpleNamespace(type="output_text", text="Hello world")
        ])
    ])
    client = FakeClient(events, final)
    provider = OpenAIProvider(client, reasoning_effort=None)
    state = ConversationState(active_provider="openai", active_model="gpt-x")
    state.messages.append(Message(role="user", content=[TextBlock(text="hi")]))

    seen = [e async for e in provider.stream_turn(state, tools=[], model="gpt-x")]

    deltas = [e for e in seen if isinstance(e, TextDelta)]
    assert "".join(d.text for d in deltas) == "Hello world"
    complete = next(e for e in seen if isinstance(e, MessageComplete))
    assert complete.message.content[0].text == "Hello world"


@pytest.mark.asyncio
async def test_function_call_streaming_produces_tool_call_start():
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="function_call", id="item_1", call_id="call_1", name="bash"),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="item_1", delta='{"command": "ls"}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(type="function_call", id="item_1", call_id="call_1", name="bash"),
        ),
    ]
    final = SimpleNamespace(output=[
        SimpleNamespace(type="function_call", call_id="call_1", name="bash", arguments='{"command": "ls"}')
    ])
    client = FakeClient(events, final)
    provider = OpenAIProvider(client, reasoning_effort=None)
    state = ConversationState(active_provider="openai", active_model="gpt-x")
    state.messages.append(Message(role="user", content=[TextBlock(text="list files")]))

    seen = [e async for e in provider.stream_turn(state, tools=[], model="gpt-x")]

    starts = [e for e in seen if isinstance(e, ToolCallStart)]
    assert starts and starts[0].tool_name == "bash" and starts[0].id == "call_1"
    complete = next(e for e in seen if isinstance(e, MessageComplete))
    tool_block = complete.message.content[0]
    assert isinstance(tool_block, ToolUseBlock)
    assert tool_block.input == {"command": "ls"}


def test_dangling_function_call_is_dropped_not_synthesized():
    """Documents the asymmetry with Claude: OpenAI just drops an
    unresolved function_call rather than needing a synthetic result."""
    provider = OpenAIProvider(client=None)
    state = ConversationState(active_provider="openai", active_model="gpt-x")
    state.messages.append(Message(role="user", content=[TextBlock(text="hi")]))
    state.messages.append(Message(
        role="assistant",
        content=[ToolUseBlock(id="call_1", tool_name="bash", input={"command": "ls"})],
        partial=True,
    ))
    # no matching tool message -- simulates an interrupt

    items = provider._to_input_items(state)
    assert all(item.get("type") != "function_call" for item in items)


def test_resolved_function_call_is_kept():
    provider = OpenAIProvider(client=None)
    state = ConversationState(active_provider="openai", active_model="gpt-x")
    state.messages.append(Message(role="user", content=[TextBlock(text="hi")]))
    state.messages.append(Message(
        role="assistant",
        content=[ToolUseBlock(id="call_1", tool_name="bash", input={"command": "ls"})],
    ))
    state.messages.append(Message(
        role="tool",
        content=[ToolResultBlock(tool_use_id="call_1", output="file.txt")],
    ))

    items = provider._to_input_items(state)
    types = [item["type"] for item in items]
    assert "function_call" in types
    assert "function_call_output" in types
