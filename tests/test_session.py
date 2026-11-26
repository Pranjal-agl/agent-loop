"""Multiplayer semantics of Session, with the mock provider."""
import asyncio

from agent_loop.core.loop import AgentLoop
from agent_loop.core.session import Session
from agent_loop.core.state import ConversationState, LoopState
from agent_loop.providers.mock_provider import MockProvider, MockStep


def make_session(script, delay_s=0.02, **kw):
    loop = AgentLoop(ConversationState(), {"mock": MockProvider(script=script, delay_s=delay_s),
                                           "other": MockProvider(canned_response="from other", delay_s=0, name="other")}, {})
    return Session(loop, session_id="t", **kw).start()


def drain_queue(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def test_late_joiner_gets_snapshot_with_history():
    s = make_session([MockStep(text="first answer")], delay_s=0)
    alice = s.subscribe("alice")
    await s.submit("hello", "alice")
    await s.wait_idle()
    bob = s.subscribe("bob")
    snap = bob.get_nowait()
    assert snap.type == "snapshot"
    assert [m["role"] for m in snap.state["messages"]] == ["user", "assistant"]
    assert snap.users == ["alice", "bob"]
    assert drain_queue(alice)[-1].type == "presence"
    await s.close()


async def test_all_subscribers_see_identical_streams():
    s = make_session([MockStep(text="a b c d")])
    qa, qb = s.subscribe("alice"), s.subscribe("bob")
    drain_queue(qa), drain_queue(qb)
    await s.submit("hi", "alice")
    await s.wait_idle()
    ea = [e.model_dump() for e in drain_queue(qa)]
    eb = [e.model_dump() for e in drain_queue(qb)]
    assert ea == eb and ea[-1]["type"] == "turn_ended"
    await s.close()


async def test_simultaneous_messages_are_queued_and_coalesced():
    s = make_session([MockStep(text="one two three four five"), MockStep(text="answering both")])
    q = s.subscribe("alice")
    await s.submit("first question", "alice")
    await asyncio.sleep(0.03)  # turn is running
    await asyncio.gather(s.submit("bob here", "bob"), s.submit("carol here", "carol"))
    await s.wait_idle()

    events = drain_queue(q)
    queued = [e for e in events if e.type == "user_message" and e.queued]
    assert {e.author for e in queued} == {"bob", "carol"}
    starts = [e for e in events if e.type == "turn_started"]
    assert len(starts) == 2  # not 3: the two queued messages became one turn
    assert set(starts[1].authors) == {"bob", "carol"}
    user_texts = [m.content[0].text for m in s.loop.state.messages if m.role == "user"]
    assert user_texts[0] == "alice: first question"
    assert "bob: bob here" in user_texts[1] and "carol: carol here" in user_texts[1]
    await s.close()


async def test_any_participant_can_interrupt_and_it_is_attributed():
    s = make_session([MockStep(text=" ".join(["word"] * 40))], delay_s=0.02)
    q = s.subscribe("alice")
    await s.submit("long answer please", "alice")
    await asyncio.sleep(0.1)
    results = await asyncio.gather(s.interrupt("bob"), s.interrupt("carol"))
    await s.wait_idle()
    assert results.count(True) >= 1
    events = drain_queue(q)
    intr = [e for e in events if e.type == "turn_interrupted"]
    assert len(intr) == 1 and intr[0].by in ("bob", "carol")
    assert intr[0].summary.kept_partial_text_chars > 0
    assert events[-1].type == "turn_ended" and events[-1].reason == "interrupted"
    assert s.loop.state.state == LoopState.IDLE
    await s.close()


async def test_interrupt_and_send_runs_new_message_next():
    s = make_session([MockStep(text=" ".join(["slow"] * 50)), MockStep(text="handled urgent")])
    q = s.subscribe("alice")
    await s.submit("long task", "alice")
    await asyncio.sleep(0.1)
    await s.submit("stop, do this instead", "bob", interrupt=True)
    await s.wait_idle()
    events = drain_queue(q)
    assert [e.type for e in events].count("turn_started") == 2
    assert s.loop.state.messages[-1].content[-1].text == "handled urgent"
    # history: user, partial assistant, interrupt note, bob's message, answer
    roles = [(m.role, m.partial, m.synthetic) for m in s.loop.state.messages]
    assert roles == [("user", False, False), ("assistant", True, False), ("user", False, True),
                     ("user", False, False), ("assistant", False, False)]
    await s.close()


async def test_concurrent_switches_are_serialized_and_broadcast():
    s = make_session([])
    q = s.subscribe("alice")
    drain_queue(q)
    await asyncio.gather(s.switch("other", "o-1", "alice"), s.switch("mock", "m-2", "bob"))
    switched = [e for e in drain_queue(q) if e.type == "provider_switched"]
    assert len(switched) == 2
    last = switched[-1]
    assert (s.loop.state.active_provider, s.loop.state.active_model) == (last.provider, last.model)
    await s.close()


async def test_unsubscribe_updates_presence():
    s = make_session([])
    qa = s.subscribe("alice")
    qb = s.subscribe("bob")
    s.unsubscribe(qb)
    pres = [e for e in drain_queue(qa) if e.type == "presence"]
    assert pres[-1].users == ["alice"] and pres[-1].left == "bob"
    await s.close()
