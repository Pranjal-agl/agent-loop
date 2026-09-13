from __future__ import annotations

import os

from pydantic import BaseModel


class Config(BaseModel):
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    default_provider: str = "mock"  # keep "mock" as default so the app
    default_model: str = "mock-1"   # runs with zero cost / zero keys
    max_iterations: int = 15

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            default_provider=os.environ.get("AGENT_PROVIDER", "mock"),
            default_model=os.environ.get("AGENT_MODEL", "mock-1"),
        )
