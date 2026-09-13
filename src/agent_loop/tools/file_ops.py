from __future__ import annotations

from pathlib import Path

from .base import ToolDefinition, ToolResult


class FileReadTool(ToolDefinition):
    name = "file_read"
    description = "Read the contents of a file."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, workdir: Path):
        self.workdir = workdir

    async def execute(self, path: str) -> ToolResult:
        target = (self.workdir / path).resolve()
        if not str(target).startswith(str(self.workdir.resolve())):
            return ToolResult(output="Refused: path escapes working directory", is_error=True)
        if not target.exists():
            return ToolResult(output=f"No such file: {path}", is_error=True)
        return ToolResult(output=target.read_text(errors="replace"))


class FileEditTool(ToolDefinition):
    name = "file_edit"
    description = "Create or overwrite a file with the given contents."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workdir: Path):
        self.workdir = workdir

    async def execute(self, path: str, content: str) -> ToolResult:
        target = (self.workdir / path).resolve()
        if not str(target).startswith(str(self.workdir.resolve())):
            return ToolResult(output="Refused: path escapes working directory", is_error=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return ToolResult(output=f"Wrote {len(content)} bytes to {path}")
