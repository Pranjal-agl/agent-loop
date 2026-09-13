"""
Live smoke test for the real provider adapters -- makes actual API calls,
so it needs real keys and costs a small amount of real money (a couple
of short calls on the cheapest model each; a fraction of a cent).

This is NOT part of the pytest suite on purpose: pytest should stay
network-free and free-to-run. Use this only when you actually have keys
and want to confirm the adapters work against the live APIs, not just
against the mocked-transport tests.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    uv run python scripts/smoke_test_live.py          # tests whichever keys are set
    uv run python scripts/smoke_test_live.py anthropic  # just one provider
    uv run python scripts/smoke_test_live.py openai
"""
from __future__ import annotations

import asyncio
import os
import sys

from agent_loop.core.state import ConversationState, Message, TextBlock
from agent_loop.providers.base import MessageComplete, TextDelta, ThinkingDelta, ToolCallStart


async def _run_and_print(provider, model: str, label: str) -> bool:
    print(f"\n--- {label} ({model}) ---")
    state = ConversationState(active_provider=label, active_model=model)
    state.messages.append(
        Message(role="user", content=[TextBlock(text="Say 'ok' and nothing else.")])
    )
    try:
        saw_complete = False
        async for event in provider.stream_turn(state, tools=[], model=model):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ThinkingDelta):
                print(f"[thinking: {event.text}]", end="", flush=True)
            elif isinstance(event, ToolCallStart):
                print(f"[tool_call_start: {event.tool_name}]")
            elif isinstance(event, MessageComplete):
                saw_complete = True
        print()
        if not saw_complete:
            print(f"FAIL ({label}): no MessageComplete event was ever yielded")
            return False
        print(f"PASS ({label})")
        return True
    except Exception as exc:
        print(f"FAIL ({label}): {exc!r}")
        return False


async def main() -> None:
    which = set(sys.argv[1:]) or {"anthropic", "openai"}
    results = []

    if "anthropic" in which:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("Skipping Anthropic: ANTHROPIC_API_KEY not set")
        else:
            from anthropic import AsyncAnthropic
            from agent_loop.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(AsyncAnthropic(api_key=key), enable_thinking=False)
            # cheapest current small model; change if this drifts out of date
            results.append(await _run_and_print(provider, "claude-haiku-4-5", "anthropic"))

    if "openai" in which:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("Skipping OpenAI: OPENAI_API_KEY not set")
        else:
            from openai import AsyncOpenAI
            from agent_loop.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(AsyncOpenAI(api_key=key), reasoning_effort=None)
            # cheapest current small model; change if this drifts out of date
            results.append(await _run_and_print(provider, "gpt-4o-mini", "openai"))

    if not results:
        print("\nNo keys set -- nothing was actually tested.")
        sys.exit(1)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
