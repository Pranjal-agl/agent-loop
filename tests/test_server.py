"""Two real WebSocket clients on one session (mock provider, no keys)."""
from fastapi.testclient import TestClient

from agent_loop.config import Config
from agent_loop.server import create_app


def recv_until(ws, kind):
    seen = []
    while True:
        ev = ws.receive_json()
        seen.append(ev)
        if ev["type"] == kind:
            return seen


def test_two_users_share_a_session_and_can_interrupt_each_other(tmp_path):
    app = create_app(tmp_path, Config())
    with TestClient(app) as client:
        assert "agent-loop" in client.get("/").text
        with client.websocket_connect("/ws/room?user=alice") as alice, \
             client.websocket_connect("/ws/room?user=bob") as bob:
            assert alice.receive_json()["type"] == "snapshot"
            assert bob.receive_json()["type"] == "snapshot"

            alice.send_json({"type": "message", "text": "!echo hi; sleep 5"})
            recv_until(bob, "tool_exec_start")       # bob sees alice's tool run live
            bob.send_json({"type": "interrupt"})     # ...and stops it
            events = recv_until(alice, "turn_ended")
            intr = next(e for e in events if e["type"] == "turn_interrupted")
            assert intr["by"] == "bob" and intr["summary"]["killed_tools"]

            bob.send_json({"type": "switch", "alias": "nope"})
            assert recv_until(bob, "notice")[-1]["text"].startswith("usage")

            bob.send_json({"type": "message", "text": "hello"})
            events = recv_until(alice, "turn_ended")
            assert events[-1]["reason"] == "completed"
            assert any(e["type"] == "user_message" and e["author"] == "bob" for e in events)

        sessions = client.get("/api/sessions").json()
        assert sessions[0]["id"] == "room" and sessions[0]["messages"] == 6
