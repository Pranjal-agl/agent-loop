"""Drives the real CLI through a pseudo-terminal: Esc and Ctrl-C must
interrupt a running tool, and the next typed message must not be lost."""
import os
import re
import select
import sys
import time
from pathlib import Path

import pytest

pty = pytest.importorskip("pty")
AGENT = Path(sys.executable).parent / "agent"


@pytest.mark.skipif(not AGENT.exists(), reason="console script not installed")
def test_esc_and_ctrl_c_interrupt_running_tool(tmp_path):
    pid, fd = pty.fork()
    if pid == 0:  # child
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env["TERM"] = "xterm"
        env["PROMPT_TOOLKIT_NO_CPR"] = "1"  # the pty never answers cursor-position requests
        os.execve(str(AGENT), [str(AGENT), "--workdir", str(tmp_path)], env)
    out = bytearray()

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                try:
                    out.extend(os.read(fd, 65536))
                except OSError:
                    return

    def plain() -> bytes:
        return re.sub(rb"\x1b\[[0-9;?]*[a-zA-Z]", b"", bytes(out))

    def wait_for(marker: bytes, count: int = 1, timeout: float = 20.0):
        end = time.time() + timeout
        while plain().count(marker) < count and time.time() < end:
            pump(0.1)
        assert plain().count(marker) >= count, f"timed out waiting for {marker!r}: {plain()[-800:]!r}"

    try:
        wait_for(b"mock-1] >")          # cold start can be slow (fresh venv)
        os.write(fd, b"!echo from-tool; sleep 30\r")
        wait_for(b"bash(echo from-tool")
        pump(0.5)
        os.write(fd, b"\x1b")            # Esc
        wait_for(b"interrupted during tool")
        pump(0.3)
        os.write(fd, b"say hi\r")        # must not be swallowed by the watcher
        wait_for(b"mock response")
        pump(0.3)
        os.write(fd, b"!sleep 30\r")
        wait_for(b"bash(sleep 30")
        pump(0.5)
        os.write(fd, b"\x03")            # Ctrl-C
        wait_for(b"interrupted during tool", 2)
        os.write(fd, b"/quit\r")
        wait_for(b"goodbye")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", out.decode(errors="replace"))
    assert text.count("interrupted during tool execution (bash)") == 2, text[-1500:]
    assert "mock response" in text, "message typed after Esc was lost"
