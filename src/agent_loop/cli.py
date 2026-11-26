"""
Terminal REPL.

Interrupts: while a turn runs, a watcher reads raw keypresses (via
prompt_toolkit's low-level Input in raw mode) and pressing Esc or Ctrl-C
calls Session.interrupt() -- during streaming, during a tool, or during a
retry backoff. A SIGINT handler does the same, so `kill -INT <pid>` works
when stdin is not a TTY. After an interrupt you're immediately back at the
prompt; the conversation continues from a reconciled history, and you can
/switch to another model first if you like.

The CLI is just a Session subscriber: exactly the code path the
multiplayer web server uses.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from .config import MODEL_ALIASES, Config, resolve_model
from .core.events import (
    ProviderSwitched, Retrying, ToolExecutionEnd, ToolExecutionStart, TurnEnded, TurnInterrupted,
)
from .core.session import Session
from .factory import build_default_loop
from .providers.base import MessageComplete, Notice, ProviderError, TextDelta, ThinkingDelta, UsageEvent

DIM, RESET, YELLOW, RED, CYAN, GREEN, BOLD = "\033[2m", "\033[0m", "\033[33m", "\033[31m", "\033[36m", "\033[32m", "\033[1m"

HELP = f"""{BOLD}Commands{RESET}
  /switch <alias>              switch model ({', '.join(MODEL_ALIASES)})
  /switch <provider> [model]   e.g. /switch anthropic claude-opus-4-5
  /model                       show current provider/model
  /history                     show the conversation as the loop stores it
  /usage                       token + cache totals
  /help, /quit
While the agent is working: {BOLD}Esc{RESET} or {BOLD}Ctrl-C{RESET} interrupts it."""


class Renderer:
    def __init__(self) -> None:
        self.mode: str | None = None  # "thinking" | "text"
        self.totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def _switch(self, mode: str | None) -> None:
        if self.mode != mode:
            if self.mode is not None:
                print(RESET)
            self.mode = mode

    def render(self, ev) -> None:
        if isinstance(ev, ThinkingDelta):
            if self.mode != "thinking":
                self._switch("thinking")
                print(f"{DIM}✻ thinking: ", end="")
            print(f"{DIM}{ev.text}", end="", flush=True)
        elif isinstance(ev, TextDelta):
            if self.mode != "text":
                self._switch("text")
            print(ev.text, end="", flush=True)
        elif isinstance(ev, ToolExecutionStart):
            self._switch(None)
            args = ev.input.get("command") or ev.input.get("path") or json.dumps(ev.input)[:120]
            extra = " (edit)" if "old_string" in ev.input else ""
            print(f"{CYAN}⏺ {ev.tool_name}{RESET}({args}){extra}")
        elif isinstance(ev, ToolExecutionEnd):
            lines = ev.output.rstrip().splitlines() or ["(no output)"]
            color = RED if ev.is_error else DIM
            shown = "\n    ".join(lines[:6])
            more = f"\n    … +{len(lines) - 6} lines" if len(lines) > 6 else ""
            print(f"{color}  ⎿ {shown}{more}  [{ev.duration_s:.1f}s]{RESET}")
        elif isinstance(ev, UsageEvent):
            self._switch(None)
            self.totals["input"] += ev.input_tokens
            self.totals["output"] += ev.output_tokens
            self.totals["cache_read"] += ev.cache_read_tokens
            self.totals["cache_write"] += ev.cache_write_tokens
            cache = f" cache read={ev.cache_read_tokens} write={ev.cache_write_tokens}" if (ev.cache_read_tokens or ev.cache_write_tokens) else ""
            reasoning = f" reasoning={ev.reasoning_tokens}" if ev.reasoning_tokens else ""
            print(f"{DIM}[{ev.model} in={ev.input_tokens} out={ev.output_tokens}{reasoning}{cache}]{RESET}")
        elif isinstance(ev, Notice):
            self._switch(None)
            print(f"{YELLOW}ℹ {ev.text}{RESET}")
        elif isinstance(ev, Retrying):
            self._switch(None)
            discarded = " (partial output above discarded)" if ev.discarded_partial else ""
            print(f"{YELLOW}↻ retry {ev.attempt} in {ev.delay_s}s: {ev.error}{discarded}{RESET}")
        elif isinstance(ev, ProviderError):
            self._switch(None)
            print(f"{RED}✗ {ev.message}{RESET}")
        elif isinstance(ev, TurnInterrupted):
            self._switch(None)
            detail = ev.summary.describe() if ev.summary else "interrupted"
            print(f"{YELLOW}⏹ {detail}{RESET}")
        elif isinstance(ev, ProviderSwitched):
            print(f"{GREEN}→ {ev.provider} / {ev.model}{RESET}")
        elif isinstance(ev, TurnEnded):
            self._switch(None)
            if ev.reason == "max_iterations":
                print(f"{YELLOW}{ev.detail}{RESET}")
            elif ev.reason == "error":
                print(f"{RED}turn failed: {ev.detail}{RESET}")
        elif isinstance(ev, MessageComplete):
            pass


async def _watch_keys(on_interrupt) -> None:
    """Runs until cancelled. Must never *return* on its own (see README:
    an early return used to be misread as a keypress)."""
    if not sys.stdin.isatty():
        await asyncio.Event().wait()
        return
    from prompt_toolkit.input import create_input
    from prompt_toolkit.keys import Keys

    inp = create_input()
    fired = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle(keys) -> None:
        for kp in keys:
            if kp.key in (Keys.ControlC, Keys.Escape) and not fired.is_set():
                fired.set()
                asyncio.ensure_future(on_interrupt())

    def ready() -> None:
        handle(inp.read_keys())
        # A lone ESC byte is ambiguous (it may start an escape sequence),
        # so the VT100 parser holds it until more input arrives. Without
        # this flush, Esc only fired on the *next* keypress -- and that
        # keypress got swallowed. Found by driving the CLI through a pty.
        loop.call_later(0.05, lambda: handle(inp.flush_keys()))

    with inp.raw_mode(), inp.attach(ready):
        await asyncio.Event().wait()


async def run_turn_ui(session: Session, q: asyncio.Queue, text: str, renderer: Renderer) -> None:
    async def interrupt() -> None:
        await session.interrupt("you")

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.ensure_future(interrupt()))
    except (NotImplementedError, RuntimeError):
        pass
    watcher = asyncio.ensure_future(_watch_keys(interrupt))
    try:
        await session.submit(text, "you")
        while True:
            ev = await q.get()
            renderer.render(ev)
            if isinstance(ev, TurnEnded):
                break
    finally:
        watcher.cancel()
        await asyncio.wait([watcher])
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError):
            pass
        renderer._switch(None)


def _print_history(session: Session) -> None:
    for i, m in enumerate(session.loop.state.messages):
        tag = m.role + (" (partial)" if m.partial else "") + (" (synthetic)" if m.synthetic else "")
        src = f" [{m.provider}/{m.model}]" if m.provider else ""
        print(f"{BOLD}{i:>3} {tag}{RESET}{DIM}{src}{RESET}")
        for b in m.content:
            body = getattr(b, "text", None) or getattr(b, "output", None) or json.dumps(getattr(b, "input", {}))
            label = b.type + (f":{b.tool_name}" if getattr(b, "tool_name", None) else "")
            print(f"      {DIM}{label}{RESET} {body.strip()[:160].replace(chr(10), ' ')}")


async def repl(session: Session) -> None:
    from prompt_toolkit import PromptSession

    renderer = Renderer()
    q = session.subscribe("you")
    q.get_nowait()  # snapshot
    while not q.empty():
        q.get_nowait()
    prompt = PromptSession() if sys.stdin.isatty() else None

    while True:
        try:
            if prompt is not None:
                st = session.loop.state
                text = await prompt.prompt_async(f"[{st.active_provider}:{st.active_model}] > ")
            else:
                text = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                if not text:
                    break
                text = text.rstrip("\n")
        except KeyboardInterrupt:
            continue  # Ctrl-C at an empty prompt just clears the line
        except EOFError:
            break
        text = text.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/help":
            print(HELP)
            continue
        if text == "/model":
            st = session.loop.state
            print(f"{st.active_provider} / {st.active_model}  (available: {', '.join(session.loop.providers)})")
            continue
        if text == "/history":
            _print_history(session)
            continue
        if text == "/usage":
            print(renderer.totals)
            continue
        if text.startswith("/switch"):
            try:
                provider, model = resolve_model(text.split()[1:])
                await session.switch(provider, model, by="you")
            except ValueError as exc:
                print(f"{RED}{exc}{RESET}")
            while not q.empty():
                renderer.render(q.get_nowait())
            continue
        await run_turn_ui(session, q, text, renderer)


def main() -> None:
    ap = argparse.ArgumentParser(prog="agent", description="Multi-provider agent loop")
    ap.add_argument("--workdir", default=".", help="directory the tools operate in (default: cwd)")
    ap.add_argument("--model", help="alias or provider:model to start with, e.g. opus or openai:gpt-5.5")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    loop = build_default_loop(workdir, Config.from_env())
    if args.model:
        spec = args.model.split(":", 1) if ":" in args.model else [args.model]
        loop.switch_provider(*resolve_model(spec))

    print(f"{BOLD}agent-loop{RESET}  providers={list(loop.providers)}  workdir={workdir}")
    print(f"{DIM}{loop.state.active_provider} / {loop.state.active_model} — /help for commands, Esc or Ctrl-C interrupts{RESET}")

    async def amain() -> None:
        session = Session(loop, attribution=False).start()
        try:
            await repl(session)
        finally:
            await session.close()

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass
    print("\ngoodbye")


if __name__ == "__main__":
    main()
