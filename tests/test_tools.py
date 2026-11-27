import asyncio
import time

import pytest

from agent_loop.tools.bash import BashTool
from agent_loop.tools.file_ops import FileEditTool, FileReadTool


async def test_file_edit_create_overwrite_and_replace(tmp_path):
    edit, read = FileEditTool(tmp_path), FileReadTool(tmp_path)
    r = await edit.execute(path="pkg/mod.py", content="x = 1\ny = 2\nx = 1\n")
    assert not r.is_error and r.output.startswith("Created")
    r = await edit.execute(path="pkg/mod.py", old_string="y = 2", new_string="y = 3")
    assert not r.is_error
    r = await edit.execute(path="pkg/mod.py", old_string="x = 1", new_string="x = 9")
    assert r.is_error and "matches 2 places" in r.output
    r = await edit.execute(path="pkg/mod.py", old_string="x = 1", new_string="x = 9", replace_all=True)
    assert "2 occurrence" in r.output
    r = await edit.execute(path="pkg/mod.py", old_string="nope", new_string="z")
    assert r.is_error and "not found" in r.output
    assert (await read.execute(path="pkg/mod.py")).output == "x = 9\ny = 3\nx = 9\n"
    r = await edit.execute(path="pkg/mod.py", content="new\n")
    assert r.output.startswith("Overwrote")
    assert not list((tmp_path / "pkg").glob(".*.tmp")), "atomic-write temp file left behind"


async def test_path_escape_uses_real_containment_not_string_prefix(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    sibling = tmp_path / "work2"
    sibling.mkdir()
    (sibling / "secret").write_text("s")
    read = FileReadTool(work)
    # a string-prefix check would accept /.../work2 because it starts with /.../work
    r = await read.execute(path="../work2/secret")
    assert r.is_error and "outside" in r.output
    (work / "link").symlink_to(sibling / "secret")
    assert (await read.execute(path="link")).is_error


async def test_file_read_offset_limit_dir_and_binary(tmp_path):
    (tmp_path / "big.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02")
    read = FileReadTool(tmp_path)
    r = await read.execute(path="big.txt", offset=3, limit=2)
    assert r.output.startswith("line3\nline4\n") and "offset=5" in r.output
    assert (await read.execute(path=".")).is_error
    assert "binary" in (await read.execute(path="bin.dat")).output


async def test_bash_reports_exit_code_and_stderr(tmp_path):
    r = await BashTool(tmp_path).execute(command="echo out; echo err >&2; exit 3")
    assert r.is_error and "out" in r.output and "err" in r.output and "[exit code 3]" in r.output


async def test_bash_timeout_kills_command(tmp_path):
    start = time.monotonic()
    r = await BashTool(tmp_path).execute(command="echo before; sleep 10", timeout=0.5)
    assert r.is_error and "timed out" in r.output and "before" in r.output
    assert time.monotonic() - start < 4


async def test_bash_does_not_read_stdin(tmp_path):
    r = await BashTool(tmp_path).execute(command="read x; echo got:$x", timeout=5)
    assert "got:" in r.output  # EOF immediately instead of hanging on the terminal


async def test_bash_cancellation_returns_partial_output(tmp_path):
    task = asyncio.ensure_future(BashTool(tmp_path).execute(command="echo partial; sleep 10"))
    await asyncio.sleep(0.4)
    task.cancel()
    result = await task  # returns instead of raising: partial output is valuable
    assert "partial" in result.output and "interrupted" in result.output


@pytest.mark.parametrize("command", [
    "rm -rf /", "rm -fr /", "rm -rf ~", ":(){ :|:& };:", "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda", "shutdown -h now",
])
async def test_destructive_commands_refused(tmp_path, command):
    result = await BashTool(tmp_path).execute(command=command)
    assert result.is_error and "Refused" in result.output


async def test_rm_rf_on_subdirectory_is_not_blocked(tmp_path):
    (tmp_path / "scratch").mkdir()
    result = await BashTool(tmp_path).execute(command="rm -rf scratch")
    assert not result.is_error
