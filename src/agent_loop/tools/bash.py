from __future__ import annotations

import asyncio
import os
import re
import signal
from pathlib import Path

from .base import ToolDefinition, ToolResult

# Best-effort guard against obviously destructive commands, NOT a sandbox
# (documented trade-off, see README). It catches accidents, not malice.
_DENYLIST_PATTERNS = [
    re.compile(r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+(/|~|\$HOME)(\s|$)"),  # rm -rf / or ~
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),               # fork bomb
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\s+.*of=/dev/"),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bchmod\s+-R\s+000\s+/"),
]

MAX_CAPTURE_BYTES = 2_000_000


class BashTool(ToolDefinition):
    name = "bash"
    description = (
        "Execute a shell command (bash) in the working directory and return combined "
        "stdout/stderr plus the exit code. Commands run non-interactively (stdin is "
        "closed). Optional `timeout` in seconds (default 120, max 600)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {"type": "number", "description": "Timeout in seconds (default 120, max 600)."},
        },
        "required": ["command"],
    }

    def __init__(self, workdir: Path, timeout_s: float = 120.0):
        self.workdir = workdir
        self.timeout_s = timeout_s

    async def execute(self, command: str, timeout: float | None = None) -> ToolResult:
        for pattern in _DENYLIST_PATTERNS:
            if pattern.search(command):
                return ToolResult(
                    output=f"Refused: command matched a destructive-command guard ({pattern.pattern})",
                    is_error=True,
                )
        limit = min(float(timeout or self.timeout_s), 600.0)

        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=self.workdir,
            stdin=asyncio.subprocess.DEVNULL,      # never steal the CLI's TTY / hang on input
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,      # interleaved, like a terminal
            start_new_session=True,                # own process group -> we can kill children too
        )
        buf = bytearray()

        async def pump() -> None:
            assert proc.stdout is not None
            while chunk := await proc.stdout.read(65536):
                if len(buf) < MAX_CAPTURE_BYTES:
                    buf.extend(chunk[: MAX_CAPTURE_BYTES - len(buf)])
            await proc.wait()

        reader = asyncio.ensure_future(pump())
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=limit)
        except asyncio.TimeoutError:
            await _kill_group(proc, reader)
            return ToolResult(output=_fmt(buf, None) + f"\n[timed out after {limit:.0f}s; process killed]", is_error=True)
        except asyncio.CancelledError:
            # Interrupted by the user: kill the whole process group (a bare
            # proc.kill() would orphan `sleep` in `cd x && sleep 100`), then
            # *return* what it printed so far -- the loop shows it to the
            # model as the killed call's partial output.
            await _kill_group(proc, reader)
            return ToolResult(output=_fmt(buf, None) + "\n[interrupted; process killed]", is_error=True)

        return ToolResult(output=_fmt(buf, proc.returncode), is_error=proc.returncode != 0)


def _fmt(buf: bytearray, returncode: int | None) -> str:
    text = buf.decode(errors="replace")
    if len(buf) >= MAX_CAPTURE_BYTES:
        text += "\n[output capture limit reached]"
    if returncode is not None:
        text = (text.rstrip("\n") + f"\n[exit code {returncode}]").lstrip("\n")
    return text


async def _kill_group(proc: asyncio.subprocess.Process, reader: asyncio.Future) -> None:
    for sig, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
        if proc.returncode is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=grace)
        except asyncio.TimeoutError:
            continue
    # Let the reader drain whatever is left in the pipe.
    try:
        await asyncio.wait_for(asyncio.shield(reader), timeout=1.0)
    except (asyncio.TimeoutError, Exception):
        reader.cancel()
