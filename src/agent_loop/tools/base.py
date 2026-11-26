from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    output: str
    is_error: bool = False


class ToolDefinition(ABC):
    """Base class for the three required tools (file_read, file_edit, bash)
    and any future ones. Providers convert this to their own function/tool
    schema at the adapter boundary -- tools themselves stay provider-agnostic."""

    name: str
    description: str
    # JSON Schema for the tool's input.
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Must be cancellable. On cancellation a tool should stop its
        work (e.g. kill its subprocess) and may *return* a ToolResult with
        whatever partial output it has, instead of re-raising: the loop
        reports that partial output to the model."""
        raise NotImplementedError

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def truncate_output(text: str, limit: int) -> str:
    """Keep head and tail of oversized tool output. Long agent loops die
    by context bloat long before they die by iteration count; an
    unbounded `cat` of a log file is the usual culprit."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n[... {omitted} characters omitted ...]\n\n{text[-tail:]}"


def resolve_in_workdir(workdir: Path, path: str) -> Path | None:
    """Resolve `path` (relative or absolute) and return it only if it is
    inside `workdir` after following symlinks. Uses is_relative_to rather
    than a string prefix check (which would let /work2 pass for /work)."""
    root = workdir.resolve()
    target = (root / path).resolve()
    return target if target == root or target.is_relative_to(root) else None
