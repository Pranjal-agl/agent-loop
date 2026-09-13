from __future__ import annotations

import asyncio
import re
from pathlib import Path

from .base import ToolDefinition, ToolResult

# Best-effort guard against obviously destructive commands, NOT a real
# sandbox -- documented tradeoff, see README. A determined user/model
# could still trivially work around this; it exists to catch accidental
# self-destructive commands, not malicious ones.
_DENYLIST_PATTERNS = [
    re.compile(r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+/(\s|$)"),  # rm -rf /
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),     # classic fork bomb
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\s+.*of=/dev/"),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bchmod\s+-R\s+000\s+/"),
]


class BashTool(ToolDefinition):
    name = "bash"
    description = "Execute a shell command in the working directory and return stdout/stderr."
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self, workdir: Path, timeout_s: float = 30.0):
        self.workdir = workdir
        self.timeout_s = timeout_s

    async def execute(self, command: str) -> ToolResult:
        for pattern in _DENYLIST_PATTERNS:
            if pattern.search(command):
                return ToolResult(
                    output=f"Refused: command matched a destructive-command guard ({pattern.pattern})",
                    is_error=True,
                )

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_s
            )
        except asyncio.CancelledError:
            # Interrupted mid-execution: kill the child process so it
            # doesn't keep running in the background, then re-raise so
            # the loop's state machine sees the cancellation.
            proc.kill()
            await proc.wait()
            raise
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(output=f"Command timed out after {self.timeout_s}s", is_error=True)

        output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        return ToolResult(output=output, is_error=proc.returncode != 0)
