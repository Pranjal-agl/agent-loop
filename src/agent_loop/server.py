"""
Multiplayer server: FastAPI + one WebSocket per client.

  uv run agent-server --workdir ./sandbox          # http://localhost:8000
  open http://localhost:8000/?session=demo&user=alice
  open http://localhost:8000/?session=demo&user=bob    (second tab)

Each session id maps to one Session (one AgentLoop, one shared history,
one shared working directory). All protocol messages are JSON:

  client -> server
    {"type": "message", "text": "...", "interrupt": false}
    {"type": "interrupt"}
    {"type": "sync"}                       -> a fresh Snapshot for this client
    {"type": "switch", "provider": "anthropic", "model": "claude-opus-5-5"}   (or {"alias": "opus"})
  server -> client
    every core.events / providers.base event, model_dump()'d, e.g.
    {"type": "text_delta", "text": "..."}, {"type": "turn_interrupted", "by": "bob", ...}
"""
from __future__ import annotations

import argparse
import asyncio
import os
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .config import Config, resolve_model
from .core.session import Session
from .factory import build_default_loop
from .providers.base import Notice


def create_app(workdir: Path, config: Config | None = None) -> FastAPI:
    app = FastAPI(title="agent-loop")
    sessions: dict[str, Session] = {}
    config = config or Config.from_env()

    def get_session(sid: str) -> Session:
        if sid not in sessions:
            sessions[sid] = Session(build_default_loop(workdir, config), session_id=sid, attribution=True).start()
        return sessions[sid]

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return resources.files("agent_loop").joinpath("web/index.html").read_text()

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict]:
        return [{"id": s.id, "users": s.users, "busy": s.busy, "messages": len(s.loop.state.messages)}
                for s in sessions.values()]

    @app.websocket("/ws/{sid}")
    async def ws(websocket: WebSocket, sid: str, user: str = "anon") -> None:
        await websocket.accept()
        session = get_session(sid)
        q = session.subscribe(user)

        async def pump() -> None:
            while True:
                ev = await q.get()
                await websocket.send_json(ev.model_dump(mode="json"))

        sender = asyncio.ensure_future(pump())
        try:
            while True:
                msg = await websocket.receive_json()
                kind = msg.get("type")
                try:
                    if kind == "message" and str(msg.get("text", "")).strip():
                        await session.submit(msg["text"].strip(), author=user, interrupt=bool(msg.get("interrupt")))
                    elif kind == "sync":  # client wants the current history (e.g. after a turn)
                        q.put_nowait(session.snapshot())
                    elif kind == "interrupt":
                        await session.interrupt(by=user)
                    elif kind == "switch":
                        spec = [msg["alias"]] if msg.get("alias") else [msg["provider"], msg["model"]]
                        provider, model = resolve_model(spec)
                        await session.switch(provider, model, by=user)
                except (ValueError, KeyError) as exc:
                    q.put_nowait(Notice(text=str(exc)))
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            session.unsubscribe(q)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="agent-server")
    ap.add_argument("--workdir", default=os.environ.get("AGENT_WORKDIR", "."))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"agent-loop server: http://{args.host}:{args.port}/?session=demo&user=alice   workdir={workdir}")
    uvicorn.run(create_app(workdir), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
