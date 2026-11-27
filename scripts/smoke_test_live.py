"""
Live smoke test against the real APIs (costs a few cents). Not part of
pytest on purpose: the test suite stays network-free.

For every provider with a key set, it runs the scenarios the interview
demo depends on, through the real AgentLoop:

  1. tool loop         -- "read secret.txt and tell me the number" (file_read)
  2. interrupt stream  -- ask for a long essay, interrupt after ~2s, continue
  3. interrupt tool    -- ask it to run `sleep 30`, interrupt mid-tool, continue
  4. prompt caching    -- two calls over a large system prompt; expects cache reads
and, if 2+ providers are configured:
  5. cross-provider    -- one conversation hopping across every provider

Usage:
    export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
    uv run python scripts/smoke_test_live.py                  # all configured providers
    uv run python scripts/smoke_test_live.py anthropic        # just one
    uv run python scripts/smoke_test_live.py --model anthropic=claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from agent_loop.config import DEFAULT_MODELS, Config
from agent_loop.core.events import ToolExecutionStart, TurnEnded
from agent_loop.factory import build_default_loop
from agent_loop.providers.base import ProviderError, UsageEvent

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


async def run(loop, text, interrupt_when=None):
    """Run one turn; `interrupt_when(event, elapsed)` -> True interrupts."""
    events = []

    async def consume():
        async for e in loop.run_turn(text):
            events.append(e)

    task = asyncio.ensure_future(consume())
    start = asyncio.get_running_loop().time()
    while not task.done():
        await asyncio.sleep(0.1)
        if interrupt_when and events and interrupt_when(events, asyncio.get_running_loop().time() - start):
            await loop.interrupt()
            break
    await asyncio.wait([task])
    if task.done() and not task.cancelled() and task.exception():
        raise task.exception()
    return events


def check(name, ok, detail=""):
    print(f"  {GREEN + 'PASS' if ok else RED + 'FAIL'}{RESET} {name} {DIM}{detail}{RESET}")
    return ok


def errors(events):
    return [e.message for e in events if isinstance(e, ProviderError)]


async def scenarios(provider: str, model: str, workdir: Path) -> bool:
    print(f"\n=== {provider} / {model}")
    loop = build_default_loop(workdir)
    loop.switch_provider(provider, model)
    results = []

    ev = await run(loop, "Read the file secret.txt with the file_read tool and reply with just the number in it.")
    reply = loop.state.messages[-1].content[-1]
    results.append(check("tool loop", any(isinstance(e, ToolExecutionStart) for e in ev) and "4217" in getattr(reply, "text", ""),
                         str(errors(ev) or getattr(reply, "text", ""))[:120]))

    ev = await run(loop, "Write a 600-word essay about the history of the shell.",
                   interrupt_when=lambda evs, t: t > 2 and any(e.type == "text_delta" for e in evs))
    s = loop.last_interrupt
    results.append(check("interrupt mid-stream", s is not None, s.describe() if s else str(errors(ev))[:120]))
    ev = await run(loop, "Never mind. In one short sentence: what was the number in secret.txt?")
    results.append(check("continue after stream interrupt", ev and isinstance(ev[-1], TurnEnded) and ev[-1].reason == "completed",
                         str(errors(ev))[:160]))

    ev = await run(loop, "Use the bash tool to run exactly: sleep 30 && echo done",
                   interrupt_when=lambda evs, t: any(isinstance(e, ToolExecutionStart) for e in evs))
    s = loop.last_interrupt
    results.append(check("interrupt mid-tool", s is not None and bool(s.killed_tools), s.describe() if s else str(errors(ev))[:120]))
    ev = await run(loop, "Don't retry that. Reply with the word OK.")
    results.append(check("continue after tool interrupt", ev and ev[-1].reason == "completed", str(errors(ev))[:160]))

    if provider in ("anthropic", "openai", "gemini"):
        cache_loop = build_default_loop(workdir)
        cache_loop.switch_provider(provider, model)
        cache_loop.state.system_prompt += "\n\nReference notes (ignore):\n" + ("The quick brown fox jumps over the lazy dog. " * 600)
        u1 = [e for e in await run(cache_loop, "Say hi.") if isinstance(e, UsageEvent)]
        u2 = [e for e in await run(cache_loop, "Say bye.") if isinstance(e, UsageEvent)]
        read = sum(e.cache_read_tokens for e in u2)
        write = sum(e.cache_write_tokens for e in u1)
        results.append(check("prompt caching", read > 0, f"call1 write={write} call2 read={read}"))
    return all(results)


async def cross_provider(models: dict[str, str], workdir: Path) -> bool:
    print(f"\n=== cross-provider: {' -> '.join(models)}")
    loop = build_default_loop(workdir)
    ok = True
    for i, (provider, model) in enumerate(list(models.items()) * 2):
        loop.switch_provider(provider, model)
        if i == 1:
            ev = await run(loop, "Run `sleep 20` with bash.", interrupt_when=lambda evs, t: any(isinstance(e, ToolExecutionStart) for e in evs))
        else:
            ev = await run(loop, f"Turn {i}: use bash to run `echo turn{i}` then reply with its output.")
        last = ev[-1] if ev else None
        ok &= check(f"turn {i} on {provider}", last is None or last.reason in ("completed",) or loop.last_interrupt is not None,
                    str(errors(ev))[:160])
    return ok


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("providers", nargs="*")
    ap.add_argument("--model", action="append", default=[], help="provider=model override")
    args = ap.parse_args()

    config = Config.from_env()
    keys = {"anthropic": config.anthropic_api_key, "openai": config.openai_api_key, "gemini": config.gemini_api_key}
    wanted = args.providers or [p for p, k in keys.items() if k]
    models = {p: DEFAULT_MODELS[p] for p in wanted if keys.get(p)}
    for override in args.model:
        p, m = override.split("=", 1)
        models[p] = m
    if not models:
        print("No API keys set -- nothing to test. Export ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY.")
        sys.exit(1)

    workdir = Path(tempfile.mkdtemp(prefix="agent-smoke-"))
    (workdir / "secret.txt").write_text("The number is 4217.\n")
    ok = True
    for provider, model in models.items():
        ok &= await scenarios(provider, model, workdir)
    if len(models) > 1:
        ok &= await cross_provider(models, workdir)
    print(f"\n{GREEN + 'ALL PASSED' if ok else RED + 'SOME CHECKS FAILED'}{RESET}  (workdir {workdir})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
