from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .base import ToolDefinition, ToolResult, resolve_in_workdir

MAX_READ_LINES = 2000


class FileReadTool(ToolDefinition):
    name = "file_read"
    description = (
        "Read a text file from the working directory. Optionally pass a 1-based "
        f"`offset` line and a `limit` (max {MAX_READ_LINES} lines per call) for large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the working directory."},
            "offset": {"type": "integer", "description": "1-based line to start from (default 1)."},
            "limit": {"type": "integer", "description": f"Max lines to return (default {MAX_READ_LINES})."},
        },
        "required": ["path"],
    }

    def __init__(self, workdir: Path):
        self.workdir = workdir

    async def execute(self, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> ToolResult:
        target = resolve_in_workdir(self.workdir, path)
        if target is None:
            return ToolResult(output=f"Refused: {path} is outside the working directory", is_error=True)
        if not target.exists():
            return ToolResult(output=f"No such file: {path}", is_error=True)
        if target.is_dir():
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ToolResult(output=f"{path} is a directory. Entries:\n" + "\n".join(entries[:500]), is_error=True)
        data = target.read_bytes()
        if b"\x00" in data[:8192]:
            return ToolResult(output=f"{path} looks like a binary file ({len(data)} bytes); not shown.", is_error=True)
        lines = data.decode(errors="replace").splitlines(keepends=True)
        offset = max(1, int(offset))
        limit = max(1, min(int(limit), MAX_READ_LINES))
        chunk = lines[offset - 1: offset - 1 + limit]
        text = "".join(chunk)
        remaining = len(lines) - (offset - 1 + len(chunk))
        if remaining > 0:
            text += f"\n[... {remaining} more lines; call again with offset={offset + len(chunk)} ...]"
        if not text:
            text = "(empty file)" if not lines else f"(offset {offset} is past the end: file has {len(lines)} lines)"
        return ToolResult(output=text)


class FileEditTool(ToolDefinition):
    name = "file_edit"
    description = (
        "Create or modify a file. Two modes:\n"
        "1) Create/overwrite: pass `content` with the full file contents.\n"
        "2) Edit in place: pass `old_string` and `new_string`; `old_string` must match the "
        "file exactly (including whitespace) and be unique unless `replace_all` is true.\n"
        "Prefer mode 2 for changes to existing files. Writes are atomic."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the working directory."},
            "content": {"type": "string", "description": "Full new file contents (mode 1)."},
            "old_string": {"type": "string", "description": "Exact text to replace (mode 2)."},
            "new_string": {"type": "string", "description": "Replacement text (mode 2)."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (mode 2)."},
        },
        "required": ["path"],
    }

    def __init__(self, workdir: Path):
        self.workdir = workdir

    async def execute(
        self,
        path: str,
        content: str | None = None,
        old_string: str | None = None,
        new_string: str | None = None,
        replace_all: bool = False,
    ) -> ToolResult:
        target = resolve_in_workdir(self.workdir, path)
        if target is None:
            return ToolResult(output=f"Refused: {path} is outside the working directory", is_error=True)
        if target.is_dir():
            return ToolResult(output=f"{path} is a directory", is_error=True)

        if old_string is not None:
            if new_string is None:
                return ToolResult(output="new_string is required with old_string", is_error=True)
            if not target.exists():
                return ToolResult(output=f"No such file: {path} (use `content` to create it)", is_error=True)
            current = target.read_text(errors="replace")
            count = current.count(old_string) if old_string else 0
            if count == 0:
                return ToolResult(output=f"old_string not found in {path}; read the file and copy it exactly", is_error=True)
            if count > 1 and not replace_all:
                return ToolResult(
                    output=f"old_string matches {count} places in {path}; add surrounding context or set replace_all",
                    is_error=True,
                )
            updated = current.replace(old_string, new_string) if replace_all else current.replace(old_string, new_string, 1)
            _atomic_write(target, updated)
            n = count if replace_all else 1
            return ToolResult(output=f"Edited {path}: replaced {n} occurrence(s)")

        if content is None:
            return ToolResult(output="Pass either `content` or `old_string`/`new_string`", is_error=True)
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, content)
        verb = "Overwrote" if existed else "Created"
        return ToolResult(output=f"{verb} {path} ({len(content.splitlines())} lines, {len(content)} bytes)")


def _atomic_write(target: Path, text: str) -> None:
    """Write to a temp file in the same directory, then os.replace().
    This is what makes "interrupt during file_edit" safe: the file is
    either entirely old or entirely new, never half-written."""
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        if target.exists():
            os.chmod(tmp, target.stat().st_mode & 0o7777)
        os.replace(tmp, target)
    except BaseException:
        with _suppress():
            os.unlink(tmp)
        raise


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
