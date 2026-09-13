from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    output: str
    is_error: bool = False


class ToolDefinition(ABC):
    """Base class for the three required tools (file_read, file_edit, bash)
    and any future ones. Providers convert this to their own function/tool-
    schema format at the adapter boundary -- tools themselves stay
    provider-agnostic."""

    name: str
    description: str
    # JSON Schema for the tool's input, e.g.
    # {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Must be cancellable: implementers should use asyncio-native I/O
        (asyncio.create_subprocess_exec, aiofiles, etc.) so that an
        asyncio.CancelledError raised mid-execution actually interrupts
        the underlying syscall rather than blocking the event loop."""
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
