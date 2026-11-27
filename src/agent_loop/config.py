from __future__ import annotations

import os

from pydantic import BaseModel

try:  # optional: pick up keys from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# Short names for /switch. Any full model id also works
# (e.g. "/switch anthropic claude-sonnet-4-5").
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "opus": ("anthropic", "claude-opus-5-5"),
    "sonnet": ("anthropic", "claude-sonnet-5-5"),
    "haiku": ("anthropic", "claude-haiku-5-5"),
    "haiku-4.5": ("anthropic", "claude-haiku-4-5"),   # manual-thinking path
    "codex": ("openai", "gpt-5.3-codex"),
    "gpt": ("openai", "gpt-5.5"),
    "gpt-mini": ("openai", "gpt-5.4-mini"),
    "gemini": ("gemini", "gemini-3.5-flash"),
    "gemini-pro": ("gemini", "gemini-3.1-pro-preview"),
    "mock": ("mock", "mock-1"),
}

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-5.3-codex",
    "gemini": "gemini-3.5-flash",
    "mock": "mock-1",
}


class Config(BaseModel):
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    default_provider: str | None = None
    default_model: str | None = None
    max_iterations: int = 50

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            default_provider=os.environ.get("AGENT_PROVIDER"),
            default_model=os.environ.get("AGENT_MODEL"),
            max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "50")),
        )


def resolve_model(args: list[str]) -> tuple[str, str]:
    """`/switch opus`, `/switch anthropic`, `/switch anthropic claude-x`."""
    if len(args) == 1 and args[0] in MODEL_ALIASES:
        return MODEL_ALIASES[args[0]]
    if len(args) == 1 and args[0] in DEFAULT_MODELS:
        return args[0], DEFAULT_MODELS[args[0]]
    if len(args) == 2:
        return args[0], args[1]
    raise ValueError("usage: /switch <alias> | /switch <provider> [model]   aliases: " + ", ".join(MODEL_ALIASES))
