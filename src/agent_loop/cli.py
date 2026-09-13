"""
Minimal REPL with real interrupt support.

Interrupt mechanism: while a turn is streaming or running a tool, a
background watcher listens at the raw terminal level (via
prompt_toolkit's low-level Input.attach/raw_mode, independent of any
prompt_toolkit Application/session) for Ctrl-C or Esc, and cancels the
turn's task the moment it's pressed -- this is what actually satisfies
"interrupt during streaming or tool execution", not just "interrupt
between turns via a blocking input() call".

Caveat: this needs a real TTY. In this development sandbox (no TTY
attached), the watcher no-ops safely rather than raising, but it hasn't
been exercised against an actual keypress here -- test it in a real
terminal before relying on it for the demo.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from prompt_toolkit.input import create_input
from prompt_toolkit.keys import Keys

from .core.loop import AgentLoop
from .core.state import ConversationState
from .config import Config
from .providers.base import MessageComplete, TextDelta, ThinkingDelta
from .providers.mock_provider import MockProvider
from .tools.bash import BashTool
from .tools.file_ops import FileEditTool, FileReadTool

INTERRUPT_KEYS = {Keys.ControlC, Keys.Escape}


def build_default_loop(workdir: Path) -> AgentLoop:
    tools = {t.name: t for t in [
        FileReadTool(workdir), FileEditTool(workdir), BashTool(workdir),
    ]}
    config = Config.from_env()
    providers = {"mock": MockProvider()}

    if config.anthropic_api_key:
        from anthropic import AsyncAnthropic
        from .providers.anthropic_provider import AnthropicProvider
        providers["anthropic"] = AnthropicProvider(AsyncAnthropic(api_key=config.anthropic_api_key))

    if config.openai_api_key:
        from openai import AsyncOpenAI
        from .providers.openai_provider import OpenAIProvider
        providers["openai"] = OpenAIProvider(AsyncOpenAI(api_key=config.openai_api_key))

    state = ConversationState(
        active_provider=config.default_provider if config.default_provider in providers else "mock",
        active_model=config.default_model,
        max_iterations=config.max_iterations,
    )
    return AgentLoop(state, providers, tools)


async def _watch_for_interrupt_key(cancel_event: asyncio.Event) -> None:
    """Sets cancel_event the moment Ctrl-C or Esc is pressed, at the raw
    input level -- works whether or not any prompt_toolkit prompt is
    currently active, which is what lets this fire *during* streaming.

    IMPORTANT: this coroutine must only ever finish by cancel_event being
    set (real interrupt) or by being cancelled from outside (turn ended
    normally). If it returned early for any other reason -- e.g. no real
    TTY available -- the caller's asyncio.wait(FIRST_COMPLETED) would
    misread that as an interrupt request. So the no-TTY case waits on the
    same event forever instead of returning, and only external
    cancellation ends it."""
    if not sys.stdin.isatty():
        await cancel_event.wait()  # never set in this branch; ends via task.cancel()
        return

    input_obj = create_input()

    def on_input_ready() -> None:
        for key_press in input_obj.read_keys():
            if key_press.key in INTERRUPT_KEYS:
                cancel_event.set()
                return

    with input_obj.raw_mode():
        with input_obj.attach(on_input_ready):
            await cancel_event.wait()


async def run_one_turn(loop: AgentLoop, user_input: str) -> None:
    cancel_event = asyncio.Event()
    consume_task = asyncio.ensure_future(_consume(loop, user_input))
    watcher_task = asyncio.ensure_future(_watch_for_interrupt_key(cancel_event))

    done, _pending = await asyncio.wait(
        {consume_task, watcher_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if watcher_task in done and not consume_task.done():
        # Interrupt key was pressed while the turn was still running.
        await loop.handle_interrupt()
        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            print("\n[interrupted -- partial output kept in context]")

    if not watcher_task.done():
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass

    if consume_task.done() and not consume_task.cancelled():
        exc = consume_task.exception()
        if exc is not None:
            raise exc


async def _consume(loop: AgentLoop, user_input: str) -> None:
    async for event in loop.run_turn(user_input):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ThinkingDelta):
            print(f"\033[2m{event.text}\033[0m", end="", flush=True)  # dim
        elif isinstance(event, MessageComplete):
            print()
    print()


def main() -> None:
    workdir = Path.cwd()
    loop = build_default_loop(workdir)
    print(f"agent-loop ready. providers={list(loop.providers.keys())} workdir={workdir}")
    print("Commands: /switch <provider> <model>   Ctrl-C or Esc to interrupt a turn.")

    async def repl():
        while True:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input.strip():
                continue
            if user_input.startswith("/switch"):
                parts = user_input.split()
                if len(parts) != 3:
                    print("usage: /switch <provider> <model>")
                    continue
                try:
                    loop.switch_provider(parts[1], parts[2])
                    print(f"switched to {parts[1]} / {parts[2]}")
                except ValueError as e:
                    print(str(e))
                continue
            await run_one_turn(loop, user_input)

    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        pass
    print("\ngoodbye")


if __name__ == "__main__":
    main()
