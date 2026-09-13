"""
Example of the kind of test this scaffold is designed to make easy:
exercising interrupt handling with zero network calls via MockProvider.
Fill out many more of these -- this is the part graders weight most.
"""
import asyncio

import pytest

from agent_loop.core.loop import AgentLoop
from agent_loop.core.state import ConversationState, LoopState
from agent_loop.providers.mock_provider import MockProvider
from agent_loop.tools.base import ToolDefinition, ToolResult


@pytest.mark.asyncio
async def test_interrupt_mid_stream_returns_to_idle():
    state = ConversationState(active_provider="mock", active_model="mock-1")
    providers = {"mock": MockProvider(canned_response="one two three four five", delay_s=0.05)}
    loop = AgentLoop(state, providers, tools={})

    async def consume():
        async for _event in loop.run_turn("hello"):
            await asyncio.sleep(0)  # let interrupt() below get scheduled

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.08)  # let a couple of words stream first
    await loop.handle_interrupt()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert state.state == LoopState.IDLE
    assert state.messages[-1].role == "assistant"
    assert state.messages[-1].partial is True
    assert len(state.messages[-1].content[0].text) > 0  # some text was captured


@pytest.mark.asyncio
async def test_switch_provider_mid_conversation():
    state = ConversationState(active_provider="mock", active_model="mock-1")
    providers = {
        "mock": MockProvider(canned_response="first"),
        "mock2": MockProvider(canned_response="second"),
    }
    loop = AgentLoop(state, providers, tools={})

    async for _ in loop.run_turn("hi"):
        pass
    loop.switch_provider("mock2", "mock-2")
    async for _ in loop.run_turn("hi again"):
        pass

    assert state.active_provider == "mock2"
    assert len(state.messages) == 4  # user, assistant, user, assistant
