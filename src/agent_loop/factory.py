"""Builds an AgentLoop with whichever providers have API keys configured."""
from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_MODELS, Config
from .core.loop import AgentLoop
from .core.state import ConversationState
from .providers.base import AgentProvider
from .providers.mock_provider import MockProvider
from .tools.bash import BashTool
from .tools.file_ops import FileEditTool, FileReadTool


def build_providers(config: Config) -> dict[str, AgentProvider]:
    providers: dict[str, AgentProvider] = {"mock": MockProvider()}
    if config.anthropic_api_key:
        from anthropic import AsyncAnthropic
        from .providers.anthropic_provider import AnthropicProvider
        providers["anthropic"] = AnthropicProvider(AsyncAnthropic(api_key=config.anthropic_api_key))
    if config.openai_api_key:
        from openai import AsyncOpenAI
        from .providers.openai_provider import OpenAIProvider
        providers["openai"] = OpenAIProvider(AsyncOpenAI(api_key=config.openai_api_key))
    if config.gemini_api_key:
        try:
            from google import genai
            from .providers.gemini_provider import GeminiProvider
            providers["gemini"] = GeminiProvider(genai.Client(api_key=config.gemini_api_key))
        except ImportError:
            pass  # `pip install agent-loop[gemini]`
    return providers


def build_default_loop(workdir: Path, config: Config | None = None) -> AgentLoop:
    config = config or Config.from_env()
    workdir = workdir.resolve()
    tools = {t.name: t for t in [FileReadTool(workdir), FileEditTool(workdir), BashTool(workdir)]}
    providers = build_providers(config)

    provider = config.default_provider
    if provider not in providers:
        # Prefer a real provider when a key is present.
        provider = next((p for p in ("anthropic", "openai", "gemini") if p in providers), "mock")
    model = config.default_model if config.default_provider == provider and config.default_model else DEFAULT_MODELS[provider]

    state = ConversationState(active_provider=provider, active_model=model, max_iterations=config.max_iterations)
    state.system_prompt += f"\nWorking directory: {workdir}"
    return AgentLoop(state, providers, tools)
