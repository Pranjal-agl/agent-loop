"""
Interrupt handling at every point of a turn, with zero network calls.
Each test ends by checking the conversation can continue cleanly.
"""
import asyncio

import pytest
from conftest import drain, run_and_interrupt

from agent_loop.core.events import ToolExecutionEnd, TurnEnded
from agent_loop.core.state import LoopState, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from agent_loop.providers.mock_provider import MockProvider, MockStep


async def assert_can_continue(loop):
    """After any interrupt, a fresh turn must complete normally."""
    loop.providers["mock"] = MockProvider(script=[MockStep(text="recovered fine")], delay_s=0)
    loop.switch_provider("mock", "mock-1")
    events = await drain(loop.run_turn("are you ok?"))
    assert isinstance(events[-1], TurnEnded) and events[-1].reason == "completed"
    assert loop.state.state == LoopState.DONE
    assert loop.state.messages[-1].content[-1].text == "recovered fine"


async def test_interrupt_mid_text_stream_keeps_partial_text(make_loop):
    loop = make_loop({"mock": MockProvider(script=[MockStep(text="one two three four five six seven")], delay_s=0.03)})
    interrupted, _, task = await run_and_interrupt(loop, "hello", 0.1)

    assert interrupted and task.cancelled()
    assert loop.state.state == LoopState.IDLE
    partial = loop.state.messages[-2]
    assert partial.role == "assistant" and partial.partial
    assert partial.content[0].text.startswith("one")
    assert "seven" not in partial.content[0].text
    note = loop.state.messages[-1]
    assert note.role == "user" and note.synthetic and "interrupted" in note.content[0].text
    assert loop.last_interrupt.kept_partial_text_chars > 0
    await assert_can_continue(loop)


async def test_interrupt_during_thinking_discards_unsigned_thinking(make_loop):
    long_thought = " ".join(["hmm"] * 50)
    loop = make_loop({"mock": MockProvider(script=[MockStep(thinking=long_thought, text="answer")], delay_s=0.02)})
    await run_and_interrupt(loop, "hi", 0.1)

    roles = [m.role for m in loop.state.messages]
    assert roles == ["user", "user"]  # no assistant message: nothing replayable was produced
    assert not any(isinstance(b, ThinkingBlock) for m in loop.state.messages for b in m.content)
    assert any("thinking" in d for d in loop.last_interrupt.discarded)
    await assert_can_continue(loop)


async def test_interrupt_during_bash_kills_process_group_and_keeps_partial_output(make_loop, tmp_path):
    marker = tmp_path / "should_not_exist"
    cmd = f"echo started; (sleep 1.5; touch {marker}) & sleep 1.5; touch {marker}"
    loop = make_loop({"mock": MockProvider(script=[MockStep(tool_calls=[("bash", {"command": cmd})])], delay_s=0)})
    await run_and_interrupt(loop, "run it", 0.4)

    tool_msg = loop.state.messages[2]
    assert tool_msg.role == "tool"
    result = tool_msg.content[0]
    assert result.is_error and "Interrupted by the user" in result.output
    assert "started" in result.output  # partial output survives
    assert loop.last_interrupt.killed_tools
    await asyncio.sleep(2.0)
    assert not marker.exists(), "background child of the killed command kept running"
    await assert_can_continue(loop)


async def test_interrupt_between_multiple_tool_calls(make_loop, tmp_path):
    calls = [
        ("file_edit", {"path": "a.txt", "content": "A"}),
        ("bash", {"command": "sleep 5"}),
        ("file_edit", {"path": "c.txt", "content": "C"}),
    ]
    loop = make_loop({"mock": MockProvider(script=[MockStep(tool_calls=calls)], delay_s=0)})
    await run_and_interrupt(loop, "do three things", 0.4)

    results = loop.state.messages[2].content
    assert [r.output.split(" ")[0] for r in results] == ["Created", "Interrupted", "Not"]
    assert (tmp_path / "a.txt").exists() and not (tmp_path / "c.txt").exists()
    s = loop.last_interrupt
    assert len(s.completed_tools) == len(s.killed_tools) == len(s.skipped_tools) == 1
    await assert_can_continue(loop)


async def test_interrupt_during_retry_backoff(make_loop):
    provider = MockProvider(script=[MockStep(error="overloaded", recoverable=True)], delay_s=0)
    loop = make_loop({"mock": provider})
    loop.retry_base_delay_s = 5.0
    interrupted, events, _ = await run_and_interrupt(loop, "hi", 0.2)
    assert interrupted
    assert any(e.type == "retrying" for e in events)
    assert loop.state.state == LoopState.IDLE
    await assert_can_continue(loop)


async def test_interrupt_when_idle_or_done_is_a_noop(make_loop):
    loop = make_loop({"mock": MockProvider(script=[MockStep(text="done")], delay_s=0)})
    assert await loop.interrupt() is False
    await drain(loop.run_turn("hi"))
    assert loop.state.state == LoopState.DONE
    n = len(loop.state.messages)
    assert await loop.interrupt() is False  # "between turns": nothing to cancel, nothing changes
    assert loop.state.state == LoopState.DONE and len(loop.state.messages) == n


async def test_concurrent_interrupts_are_idempotent(make_loop):
    loop = make_loop({"mock": MockProvider(script=[MockStep(text="a b c d e f g h")], delay_s=0.05)})
    task = asyncio.ensure_future(drain(loop.run_turn("hi")))
    await asyncio.sleep(0.1)
    results = await asyncio.gather(loop.interrupt(), loop.interrupt(), loop.interrupt())
    await asyncio.wait([task])
    assert sum(results) >= 1
    assert loop.state.state == LoopState.IDLE
    assert sum(1 for m in loop.state.messages if m.synthetic) == 1


async def test_cancel_consumer_directly_also_reconciles(make_loop):
    """Cancelling the consuming task (e.g. a web client disconnect) goes
    through exactly the same reconciliation as interrupt()."""
    loop = make_loop({"mock": MockProvider(script=[MockStep(text="x y z w v")], delay_s=0.05)})
    task = asyncio.ensure_future(drain(loop.run_turn("hi")))
    await asyncio.sleep(0.12)
    task.cancel()
    await asyncio.wait([task])
    assert loop.state.state == LoopState.IDLE
    assert loop.state.messages[-1].synthetic
    await assert_can_continue(loop)


async def test_breaking_out_of_iteration_reconciles(make_loop):
    """GeneratorExit path: the consumer stops iterating mid-turn."""
    loop = make_loop({"mock": MockProvider(script=[MockStep(text="a b c d e f")], delay_s=0.01)})
    agen = loop.run_turn("hi")
    async for e in agen:
        if e.type == "text_delta":
            break
    await agen.aclose()
    assert loop.state.state == LoopState.IDLE
    await assert_can_continue(loop)


async def test_interrupt_then_switch_provider_gets_valid_history(make_loop):
    first = MockProvider(script=[MockStep(thinking="plan", tool_calls=[("bash", {"command": "sleep 5"})])], delay_s=0, name="mock")
    second = MockProvider(script=[MockStep(text="hello from the other model")], delay_s=0, name="other")
    loop = make_loop({"mock": first, "other": second})
    await run_and_interrupt(loop, "start", 0.3)
    loop.switch_provider("other", "other-1")
    events = await drain(loop.run_turn("never mind, say hi"))
    assert events[-1].reason == "completed"
    seen = second.calls[0].messages
    # the second provider saw: user, assistant(tool_use), tool(result), note, user
    uses = [b.id for m in seen for b in m.content if isinstance(b, ToolUseBlock)]
    results = [b.tool_use_id for m in seen for b in m.content if isinstance(b, ToolResultBlock)]
    assert uses == results


async def test_switch_provider_mid_conversation():
    from agent_loop.core.loop import AgentLoop
    from agent_loop.core.state import ConversationState
    state = ConversationState(active_provider="mock", active_model="mock-1")
    loop = AgentLoop(state, {"mock": MockProvider(canned_response="first", delay_s=0),
                             "mock2": MockProvider(canned_response="second", delay_s=0)}, tools={})
    await drain(loop.run_turn("hi"))
    loop.switch_provider("mock2", "mock-2")
    await drain(loop.run_turn("hi again"))
    assert state.active_provider == "mock2"
    assert [m.role for m in state.messages] == ["user", "assistant", "user", "assistant"]
    assert state.messages[-1].model == "mock-2"
    with pytest.raises(ValueError):
        loop.switch_provider("nope", "x")


# --- long-running loops, errors, tool failures --------------------------------

async def test_long_tool_loop_and_max_iterations(make_loop):
    steps = [MockStep(tool_calls=[("bash", {"command": f"echo {i}"})]) for i in range(10)]
    loop = make_loop({"mock": MockProvider(script=steps + [MockStep(text="finally")], delay_s=0)})
    loop.state.max_iterations = 5
    events = await drain(loop.run_turn("loop a lot"))
    assert events[-1].reason == "max_iterations"
    assert loop.state.state == LoopState.DONE  # not wedged
    assert loop.state.messages[-1].role == "tool"  # history ends valid
    # "continue" picks up where it left off
    loop.state.max_iterations = 50
    events = await drain(loop.run_turn("continue"))
    assert events[-1].reason == "completed"
    assert sum(1 for e in events if isinstance(e, ToolExecutionEnd)) == 5


async def test_transient_errors_are_retried(make_loop):
    script = [MockStep(text="partial words here", error="529 overloaded", recoverable=True, error_after_words=2),
              MockStep(text="success")]
    loop = make_loop({"mock": MockProvider(script=script, delay_s=0)})
    events = await drain(loop.run_turn("hi"))
    retry = next(e for e in events if e.type == "retrying")
    assert retry.discarded_partial
    assert events[-1].reason == "completed"
    texts = [b.text for m in loop.state.messages for b in m.content if isinstance(b, TextBlock)]
    assert texts == ["hi", "success"]  # failed attempt's partial text never entered history


async def test_fatal_error_ends_turn_and_conversation_continues(make_loop):
    loop = make_loop({"mock": MockProvider(script=[MockStep(error="400 invalid request", recoverable=False)], delay_s=0)})
    events = await drain(loop.run_turn("hi"))
    assert events[-1].reason == "error"
    assert loop.state.state == LoopState.IDLE
    await assert_can_continue(loop)


async def test_tool_failures_are_reported_not_raised(make_loop):
    calls = [("nonexistent_tool", {}), ("file_read", {"wrong_arg": 1}),
             ("bash", {"__raw_arguments__": "{not json"}), ("file_read", {"path": "../../etc/passwd"})]
    loop = make_loop({"mock": MockProvider(script=[MockStep(tool_calls=calls), MockStep(text="ok")], delay_s=0)})
    events = await drain(loop.run_turn("break things"))
    assert events[-1].reason == "completed"
    results = loop.state.messages[2].content
    assert all(r.is_error for r in results)
    assert "Unknown tool" in results[0].output
    assert "Invalid arguments" in results[1].output
    assert "not valid JSON" in results[2].output
    assert "outside the working directory" in results[3].output


async def test_huge_tool_output_is_truncated(make_loop):
    loop = make_loop({"mock": MockProvider(script=[MockStep(tool_calls=[("bash", {"command": "seq 1 200000"})]),
                                                   MockStep(text="ok")], delay_s=0)}, max_tool_output_chars=5000)
    await drain(loop.run_turn("spam"))
    out = loop.state.messages[2].content[0].output
    assert len(out) < 5200 and "characters omitted" in out and out.rstrip().endswith("[exit code 0]")


async def test_cannot_start_turn_while_busy(make_loop):
    loop = make_loop({"mock": MockProvider(script=[MockStep(text="a b c d")], delay_s=0.05)})
    task = asyncio.ensure_future(drain(loop.run_turn("one")))
    await asyncio.sleep(0.05)
    with pytest.raises(RuntimeError):
        await drain(loop.run_turn("two"))
    await asyncio.wait([task])
