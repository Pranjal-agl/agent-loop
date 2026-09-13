import pytest

from agent_loop.tools.bash import BashTool


@pytest.mark.asyncio
async def test_normal_command_runs(tmp_path):
    tool = BashTool(tmp_path)
    result = await tool.execute(command="echo hello")
    assert not result.is_error
    assert "hello" in result.output


@pytest.mark.parametrize("command", [
    "rm -rf /",
    "rm -fr /",
    ":(){ :|:& };:",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    "shutdown -h now",
])
@pytest.mark.asyncio
async def test_destructive_commands_refused(tmp_path, command):
    tool = BashTool(tmp_path)
    result = await tool.execute(command=command)
    assert result.is_error
    assert "Refused" in result.output


@pytest.mark.asyncio
async def test_rm_rf_on_subdirectory_is_not_blocked(tmp_path):
    """The denylist targets root-level destruction specifically -- normal
    cleanup within the working directory should still work."""
    (tmp_path / "scratch").mkdir()
    tool = BashTool(tmp_path)
    result = await tool.execute(command="rm -rf scratch")
    assert not result.is_error
