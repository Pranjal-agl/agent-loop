import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from agent_loop.core.loop import AgentLoop  # noqa: E402
from agent_loop.core.state import ConversationState  # noqa: E402
from agent_loop.tools.bash import BashTool  # noqa: E402
from agent_loop.tools.file_ops import FileEditTool, FileReadTool  # noqa: E402


@pytest.fixture
def tools(tmp_path):
    return {t.name: t for t in [FileReadTool(tmp_path), FileEditTool(tmp_path), BashTool(tmp_path)]}


@pytest.fixture
def make_loop(tools):
    def _make(providers, active="mock", **kw):
        state = ConversationState(active_provider=active, active_model=f"{active}-model")
        return AgentLoop(state, providers, tools, retry_base_delay_s=0.01, **kw)
    return _make


async def drain(agen):
    return [e async for e in agen]


async def run_and_interrupt(loop, text, after_s):
    """Run a turn in its own task and interrupt it after `after_s`."""
    events = []

    async def consume():
        async for e in loop.run_turn(text):
            events.append(e)

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(after_s)
    interrupted = await loop.interrupt()
    await asyncio.wait([task])
    return interrupted, events, task
