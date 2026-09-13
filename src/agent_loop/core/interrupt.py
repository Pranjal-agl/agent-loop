from __future__ import annotations

import asyncio


class InterruptManager:
    """Owns the currently-running asyncio.Task for a conversation (the
    provider stream, or a tool execution) so that `interrupt()` can be
    called from a different coroutine (e.g. the CLI's input-reading task)
    without the two racing on shared state."""

    def __init__(self) -> None:
        self._active_task: asyncio.Task | None = None

    def track(self, task: asyncio.Task) -> None:
        self._active_task = task

    def clear(self) -> None:
        self._active_task = None

    async def interrupt(self) -> bool:
        """Returns True if something was actually cancelled."""
        if self._active_task is None or self._active_task.done():
            return False
        self._active_task.cancel()
        try:
            await self._active_task
        except asyncio.CancelledError:
            pass
        return True
