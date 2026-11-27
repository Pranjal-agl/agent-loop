"""
Session: the single owner of an AgentLoop, shared by any number of
clients (one CLI, or many browser tabs / users in multiplayer mode).

Concurrency model
-----------------
* One worker task drains an inbox and runs at most one turn at a time.
  There are no data races on ConversationState because only the worker's
  turn task mutates it (plus `switch`, which is serialized by a lock and
  only flips two fields that are read at the start of each model call).
* Messages that arrive while a turn is running are QUEUED, not barged in:
  silently cutting off someone else's request because you typed second
  is the worse failure mode. When the turn ends, everything queued is
  *coalesced* into one user turn ("alice: ...\\nbob: ..."), so the agent
  answers both people at once instead of running two stale turns back to
  back.
* Interrupting is an explicit action (Stop button, Esc, or "send &
  interrupt"), available to any participant, and attributed ("bob
  interrupted"). It is idempotent, so two people pressing Stop together
  is harmless.
* Every event is broadcast to every subscriber, so all clients render
  the same stream in real time. Late joiners get a Snapshot first.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import Counter

from .events import (
    Presence, ProviderSwitched, Snapshot, TurnEnded, TurnInterrupted, TurnStarted, UserMessageEvent,
)
from .loop import AgentLoop


class Session:
    def __init__(self, loop: AgentLoop, session_id: str | None = None, attribution: bool = True):
        self.id = session_id or uuid.uuid4().hex[:8]
        self.loop = loop
        # Prefix user text with "author: " so the model knows who said what.
        # Off for the single-user CLI.
        self.attribution = attribution
        self._inbox: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._subscribers: dict[asyncio.Queue, str] = {}
        self._users: Counter[str] = Counter()
        self._worker: asyncio.Task | None = None
        self._current: asyncio.Task | None = None
        self._interrupted_by: str | None = None
        self._switch_lock = asyncio.Lock()

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "Session":
        if self._worker is None:
            self._worker = asyncio.ensure_future(self._work())
        return self

    async def close(self) -> None:
        await self.interrupt("system")
        if self._worker:
            self._worker.cancel()
            await asyncio.wait([self._worker])

    @property
    def busy(self) -> bool:
        return self._current is not None and not self._current.done()

    @property
    def users(self) -> list[str]:
        return sorted(u for u, n in self._users.items() if n > 0)

    # ------------------------------------------------------------ pub/sub

    def subscribe(self, user: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[q] = user
        self._users[user] += 1
        q.put_nowait(self.snapshot())
        self.broadcast(Presence(users=self.users, joined=user))
        return q

    def snapshot(self) -> Snapshot:
        from ..config import MODEL_ALIASES
        return Snapshot(
            session_id=self.id, state=self.loop.state.model_dump(mode="json"),
            providers=list(self.loop.providers), users=self.users, queued=self._inbox.qsize(),
            models=MODEL_ALIASES,
        )

    def unsubscribe(self, q: asyncio.Queue) -> None:
        user = self._subscribers.pop(q, None)
        if user is not None:
            self._users[user] -= 1
            self.broadcast(Presence(users=self.users, left=user))

    def broadcast(self, event) -> None:
        for q in list(self._subscribers):
            q.put_nowait(event)

    # ------------------------------------------------------------ actions

    async def submit(self, text: str, author: str = "user", interrupt: bool = False) -> None:
        """Queue a message. With interrupt=True, also stop the running turn
        so this message (plus anything else queued) runs next."""
        queued = self.busy or not self._inbox.empty()
        self._inbox.put_nowait((author, text))
        self.broadcast(UserMessageEvent(author=author, text=text, queued=queued and not interrupt))
        if interrupt:
            await self.interrupt(author)

    async def interrupt(self, by: str = "user") -> bool:
        if not self.busy:
            return False
        self._interrupted_by = by
        return await self.loop.interrupt()

    async def switch(self, provider: str, model: str, by: str = "user") -> None:
        async with self._switch_lock:
            self.loop.switch_provider(provider, model)
        self.broadcast(ProviderSwitched(by=by, provider=provider, model=model))

    async def wait_idle(self) -> None:
        """Test helper: wait until the inbox is drained and no turn runs."""
        while self.busy or not self._inbox.empty():
            await asyncio.sleep(0.01)

    # ------------------------------------------------------------ worker

    async def _work(self) -> None:
        while True:
            batch = [await self._inbox.get()]
            while not self._inbox.empty():
                batch.append(self._inbox.get_nowait())
            self._current = asyncio.ensure_future(self._run(batch))
            await asyncio.wait([self._current])  # never raises, even if the turn was cancelled

    def _compose(self, batch: list[tuple[str, str]]) -> str:
        if self.attribution:
            return "\n".join(f"{a}: {t}" for a, t in batch)
        return "\n\n".join(t for _, t in batch)

    async def _run(self, batch: list[tuple[str, str]]) -> None:
        authors = list(dict.fromkeys(a for a, _ in batch))
        self._interrupted_by = None
        self.broadcast(TurnStarted(
            authors=authors, provider=self.loop.state.active_provider, model=self.loop.state.active_model,
        ))
        try:
            async for event in self.loop.run_turn(self._compose(batch), author=", ".join(authors)):
                self.broadcast(event)
        except asyncio.CancelledError:
            # The loop already reconciled history; just tell everyone.
            self.broadcast(TurnInterrupted(by=self._interrupted_by or "system", summary=self.loop.last_interrupt))
            self.broadcast(TurnEnded(reason="interrupted", iterations=self.loop.state.iteration_count))
        except Exception as exc:  # e.g. RuntimeError from a misuse; keep the worker alive
            self.broadcast(TurnEnded(reason="error", detail=f"{type(exc).__name__}: {exc}"))
