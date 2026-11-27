# agent-loop

**Live site: https://pranjal-agl.github.io/agent-loop/**. It opens a demo agent that runs in your browser, so you can try interrupts, model switching and multiplayer with no setup. Use the connection button in the header to point the same page at your own `agent-server` and talk to the real models.

A multi-provider agent loop: **Claude, OpenAI and Gemini**, with file and shell tools. You can interrupt it at any point and the conversation continues cleanly. You can switch models between messages, and several people can share one session.

```
uv sync --all-extras
uv run pytest                    # 86 tests, no network, no keys
uv run agent --workdir ./sandbox # terminal UI (mock provider if no keys)
uv run agent-server --workdir ./sandbox
#   → http://127.0.0.1:8000/?session=demo&user=alice   (open a 2nd tab as bob)
```

Add keys (or put them in `.env`) to enable real providers. Only providers that have a key are registered.

```
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
uv run agent --model opus
> /switch codex          # aliases: opus sonnet haiku haiku-4.5 codex gpt gpt-mini gemini gemini-pro mock
> /switch anthropic claude-opus-4-5   # or any full model id
```

Before a live demo, run `uv run python scripts/smoke_test_live.py`. It runs a tool loop, a mid-stream interrupt, a mid-tool interrupt, prompt caching, and one conversation that hops across every configured provider, all against the real APIs. See [SETUP.md](SETUP.md) and the demo script at the end of this file.

---

## 1. Problem framing

The core difficulty isn't calling three SDKs. It's that **the conversation must always be a valid request for whichever provider is called next**. That has to hold even after the user cancels at an arbitrary instant, and even when the history was written by a different provider. I designed around three decisions:

| Decision | Why |
|---|---|
| **One provider-agnostic `ConversationState`** (`core/state.py`) is the only source of truth. Adapters translate it to wire format on *every* call. | Switching model is then a two-field pointer change. No provider keeps server-side state: OpenAI runs with `store=False` and Gemini needs none. |
| **One cancellation path.** Interrupting means cancelling the task that iterates `run_turn()`. | Cancellation lands at whatever `await` the turn is suspended on: the HTTP stream, a subprocess, the retry backoff, or between iterations. It is caught in exactly one place. |
| **Each layer owns its own invariants.** The loop guarantees a *semantically complete* history. Each adapter guarantees a *syntactically valid* request for its provider. | Provider quirks don't leak into the loop. The adapters' repair logic also acts as a second safety net if the loop's bookkeeping is ever wrong. |

Assumptions: one working directory per session (the tools refuse paths outside it), a trusted local user, and turns that run one at a time per session.

## 2. Architecture

```
          CLI (cli.py)            Web/WebSocket (server.py + web/index.html)
                  \                     /
                   Session  (core/session.py)   queue · fan-out · interrupt policy · attribution
                      |
                   AgentLoop (core/loop.py)     turn loop · tools · retries · reconciliation
                   /     |      \
     AnthropicProvider OpenAIProvider GeminiProvider MockProvider   (providers/*, all implement AgentProvider)
                      |
         ConversationState (core/state.py)      messages + content blocks + LoopState machine
```

- **`AgentProvider`** (`providers/base.py`) has one method that matters: `stream_turn(state, tools, model) → AsyncIterator[StreamEvent]`, ending in exactly one `MessageComplete` or `ProviderError`. The Gemini adapter was added without touching the loop, the state, or the other adapters. That's the extensibility test.
- **Content blocks** carry provider-owned opaque data: Claude `signature` / `redacted` thinking, OpenAI `provider_meta={id, encrypted_content}`, Gemini `thought_signature` on text and call blocks. `ThinkingBlock.provider` records the author. Each adapter replays only what it can read.
- **The state machine is data** (`VALID_TRANSITIONS`). An illegal transition raises, which is how several of the bugs below were caught.
- **Tools** (`tools/`): `file_read` (offset/limit, refuses binaries and directories), `file_edit` (create/overwrite, or an exact `old_string → new_string` edit, always an **atomic** temp-file + `os.replace`), and `bash` (own process group, stdin closed, timeout, exit code, output cap).

## 3. Interruption handling

You can interrupt from anywhere: Esc or Ctrl-C in the CLI, Stop or *Interrupt & send* in the web UI, or a client disconnecting. All of them cancel the turn task. `AgentLoop._reconcile()` then uses a per-turn `_TurnContext` to tell exactly what was in flight and decides as follows:

| What was in flight | What happens | Why |
|---|---|---|
| Streamed text not yet committed | Kept as an `assistant` message with `partial=True` | The user saw it, so the model should too. Otherwise the next answer repeats or contradicts it. |
| Streamed thinking | Discarded (counted in the summary) | A signature only arrives at the end of a block. Unsigned thinking can't be replayed to Claude, and it means nothing to the other providers. |
| Half-streamed tool-call arguments | Discarded | Never executed, so there is nothing to report. |
| A committed assistant message whose tools are still running | Every call gets **exactly one** result: *completed* (real output), *killed* (partial stdout + "killed after 3.2s; side effects may be partial"), or *not run*. | The model must know which side effects actually happened. Tools run **sequentially** so each one is unambiguously in one of these three states. |
| A running `bash` | SIGTERM → SIGKILL to the **whole process group**, then return its partial output | A bare `proc.kill()` orphans the children of `sh -c`. The test checks that a background child does not survive. |
| `file_edit` | Atomic, so the file is always fully old or fully new | Hard-killing a write can't leave a half-written file. |
| Retry backoff | Just cancelled | Nothing was committed. |
| Nothing (idle, or between turns) | `interrupt()` returns `False`; history and state are unchanged | It's idempotent, so two users pressing Stop together is fine. |

Then a synthetic user note goes into the history ("The user interrupted… respond to their next message"), and the state machine moves `STREAMING/TOOL_RUNNING → INTERRUPTED → IDLE`. The user sees an `InterruptSummary`, for example: `interrupted during tool execution (bash); kept 112 chars of partial text; killed: bash#1fda8b after 1.3s`.

**Why kill tools instead of letting them finish?** The scaffold let them finish. That makes interrupt latency unbounded (try `sleep 600`), and "the user said stop" should mean stop. Atomic file writes plus reporting partial output give the safety benefit of finishing without the wait.

### Provider-specific resume invariants

| | Claude | OpenAI (Responses) | Gemini |
|---|---|---|---|
| Unanswered tool call | `tool_result` required in the **immediately next** user message, tool results first → inserted **in place** | every `function_call` needs an output somewhere later (`No tool output found…`) → unanswered calls dropped | `function_response` in the next user content → synthesized in place |
| Role order | strict alternation → consecutive same-role messages merged | free-form item list, nothing to merge | alternation → contents merged (never *parts*) |
| Reasoning replay | thinking + redacted_thinking blocks verbatim with signature; foreign blocks dropped | encrypted reasoning items verbatim; a reasoning item must be followed by its output, so it's dropped if its call was dropped | `thought_signature` stays on its original part; foreign calls get the documented `skip_thought_signature_validator` |
| Mid-tool-loop switch *into* the provider | manual-thinking models (4.5 family) require the last tool-loop assistant turn to start with their own thinking → thinking is disabled for that one request, with a `Notice`; adaptive models (4.6+/5.x) relax the rule | fine | fine (sentinel) |
| Empty content | empty text and empty messages rejected → filtered | — | empty parts → filtered |

`tests/test_cross_provider.py` runs **one** conversation through Claude (adaptive), OpenAI, Gemini, Claude (manual thinking) and OpenAI again, with an interrupt mid-tool and another mid-stream. Every request each fake transport receives is checked by executable versions of the rules above (`tests/wire_validators.py`).

## 4. Provider-specific optimizations

**Claude: thinking.** The mode is chosen per model (`thinking_mode_for`):
- 4.6+ and 5.x models: `thinking: {type: "adaptive", display: "summarized"}` (optional `output_config.effort`). Interleaving between tool calls is automatic. These models reject `type: "enabled"`, and newer ones default to `display: "omitted"`, so we ask for summaries to make reasoning visible.
- 4.0–4.5: `{type: "enabled", budget_tokens}` + `anthropic-beta: interleaved-thinking-2025-05-14`, with `max_tokens > budget_tokens`. The scaffold sent 4096/4096, which the API rejects.

**Claude: prompt caching** uses explicit breakpoints, at most 4:
1. **system prompt**: the prefix order is tools → system → messages, so this one breakpoint also caches the tool definitions;
2. **the last block of the conversation**: a moving breakpoint that writes the whole history each call;
3. **the last block of the previous user turn**: where the *previous* request's breakpoint was, which guarantees a read hit even when a tool-heavy iteration adds more than the 20-block lookback window;
4. one optional user-pinned breakpoint (`state.cache_breakpoints`).

Thinking blocks can't carry `cache_control`, so the marker walks back to the nearest eligible block. Request building never mutates state, so the cached prefix stays byte-stable. `UsageEvent` shows `cache read/write` in both UIs.

**OpenAI:** Responses API (not Chat Completions), so reasoning survives across tool calls. `store=False` + `include=["reasoning.encrypted_content"]` keeps it stateless and ZDR-compatible. `reasoning.summary="auto"` streams summaries. Caching is automatic: we pin `prompt_cache_key` to the conversation id and report `cached_tokens`. Reasoning params are only sent to reasoning models (gpt-5.x, codex, o-series).

**Gemini:** thought signatures are round-tripped, implicit caching is reported, automatic function calling is disabled (the loop owns execution), and thought summaries are streamed.

**Long-running loops.** A turn iterates up to `max_iterations` (default 50, `AGENT_MAX_ITERATIONS`). Hitting the cap ends the turn *gracefully*: history stays valid and "continue" resumes. Transient errors (429, 5xx, 529 overloaded, network) retry with jittered exponential backoff. Partial output from a failed attempt is never committed, and UIs are told to strike it out. Tool output is capped at 30k characters (head + tail), because long loops run out of context long before they run out of iterations.

## 5. Multiplayer mode (implemented)

`Session` (`core/session.py`) owns one `AgentLoop`. Any number of clients subscribe, and `server.py` exposes it over WebSockets with a small web UI.

- **Two people send at the same time.** A single worker runs one turn at a time, and only that turn mutates state, so there are no data races. Messages that arrive while a turn runs are **queued, not barged in**: silently cutting off someone else's request because you typed second is the worse failure. When the turn ends, everything queued is **coalesced into one turn** (`alice: …\nbob: …`), so the agent answers both together instead of running stale turns back to back.
- **Interrupting is explicit and attributed.** Any participant can press Stop or use *Interrupt & send*, which stops the turn and runs their message next. Everyone sees "⏹ bob interrupted — killed bash after 1.3s". It reuses `AgentLoop.interrupt()` unchanged, and it's idempotent.
- **Real-time visibility.** Every event (thinking/text deltas, tool start/end, usage, retries, switches) is broadcast to every client. Presence is a separate ephemeral event that never touches history. Late joiners get a `Snapshot` of the full state.
- **Conflicts.** Model switches go through a lock and are broadcast, so the last writer wins and everyone sees it. A switch during a turn takes effect at the next model call. `Message.author` records who said what, and user text is prefixed with the author's name so the model can tell people apart.
- **Not done, and how I'd do it.** Persistence (state is already a pydantic model, so it would serialize to Postgres/Redis per session). Auth and roles (who may interrupt or switch). Per-user rate limits. Multi-process fan-out (Redis pub/sub instead of in-process queues). Bounded subscriber queues so a slow client gets dropped instead of growing memory. Per-user workdirs or a container sandbox.

### The web UI

`src/agent_loop/web/index.html` is one self-contained page. `agent-server` serves it, and GitHub Pages hosts the same file (`.github/workflows/pages.yml` republishes it on every change to `main`).
- **Trace strip:** each turn is drawn live as lanes for state, thinking, text and tools. An interrupt is a red cut line labeled with who pressed Stop.
- **Inspector:** the loop's state machine with the current state lit; the exact context the model will see next, so you can watch the partial reply and the interrupt note appear; and per-call prompt-cache bars.
- **Two modes:** on GitHub Pages it starts with a demo agent. That's a simulated model with a pretend shell and a small Python project, speaking the same WebSocket protocol and following the same interrupt and queueing rules as the Python loop. Switch to *My agent-server* to drive the real loop; Chrome may ask to allow access to your local network. Served by `agent-server`, it connects to that server automatically.
- **Multiplayer:** presence avatars, "queued" markers, coalesced turns ("answering alice and bob"), attributed interrupts, and *Interrupt and send*. In demo mode you can send as bob, or queue a message from bob while the agent works.

## 6. Bugs found and fixed

The scaffold "worked" in its own tests. Exercising interruption at every phase surfaced these:

1. **Interrupt during tool execution crashed.** `InterruptManager` only tracked the *stream* task. During tools `interrupt()` returned `False`, so the loop attempted `TOOL_RUNNING + CLEANUP_DONE`, an illegal transition, and raised. → Replaced by the single-cancellation-path design.
2. **Interrupt between turns crashed.** `DONE + CLEANUP_DONE` is illegal. → Interrupt is now a no-op when idle.
3. **The synthetic `tool_result` was appended at the end of the history.** After an interrupt *and a new user message*, Claude rejects that (the result must be in the very next message). The existing test only covered the case where the dangling call was last. → Repair now happens in place.
4. **Partial thinking was saved as visible text.** The partial-message buffer concatenated thinking and text deltas into a `TextBlock`, leaking reasoning into the transcript the model sees.
5. **Cross-provider thinking replay.** OpenAI reasoning ids were stored in the same `signature` field and would have been sent to Claude as thinking signatures (rejected), and vice versa. → Blocks are now tagged with `provider`.
6. **`max_tokens` == `budget_tokens`** (4096/4096): rejected by the API whenever thinking was on. Newer Claude models also reject `type: "enabled"` outright. → Thinking mode is chosen per model.
7. **`MaxIterationsExceeded` left the state wedged in `STREAMING`**, so every later message failed. → The turn now ends gracefully.
8. **Tool exceptions killed the turn.** Wrong arguments (`TypeError`), reading a directory, invalid JSON. → Each becomes an error result the model can react to.
9. **`bash` orphaned children** (`proc.kill()` only kills the shell) and inherited the terminal's stdin, which competed with the Esc key watcher. → Process group + `stdin=DEVNULL`.
10. **The path check used a string prefix**, so `/work2` passed for `/work`. → `Path.is_relative_to` after resolving symlinks.
11. **A lone Esc keypress fired late and ate the next message.** The VT100 parser holds a bare `ESC` byte, since it might start an escape sequence, until more input arrives. The interrupt therefore fired on the *next* keypress, and the watcher swallowed that input. Found by driving the real CLI through a pseudo-terminal (`tests/test_cli_pty.py`), which the scaffold noted it had never been able to test. → Pending keys are flushed after 50 ms.
12. `ProviderError` was swallowed silently by the CLI, and OpenAI got no system prompt.
13. **Earlier scaffold bugs, kept for the story:** cancellation translated into a normal "stream ended"; completed messages were never appended; a no-TTY watcher returned early and read as a keypress.

## 7. Trade-offs and limitations

- **Sequential tools** give up parallel speed for deterministic side effects and precise interrupt reporting. Running read-only tools concurrently would be a cheap improvement.
- **The bash denylist is not a sandbox.** It catches accidents, not malice. Real isolation would be a container or jail per session.
- **No context compaction** beyond truncating tool output. Very long sessions will eventually hit the context window. The next step is summarizing old turns (Claude's context-management features, or our own summarizer behind the same `ConversationState`).
- **Partial assistant text is kept** after interrupts. Some products drop it instead. Keeping it means the model knows what the user already saw, at the cost of occasionally leaving a dangling half-sentence in context.
- **Live API verification:** the adapters are verified against introspected SDK types and docs, with fake transports. I couldn't reach the provider APIs from the build environment. `scripts/smoke_test_live.py` is the pre-demo check.

## 8. Tests

`uv run pytest` runs 86 tests in about 15 seconds with no network:
- `test_interrupt.py`: interrupt during text, thinking, a single tool, the middle of several tools, and retry backoff; consumer cancellation and `aclose()`; idempotent concurrent interrupts; max iterations; retries; tool failures; truncation. Every interrupt test ends by checking that a new turn completes.
- `test_cross_provider.py`: the five-hop conversation above, validated against each provider's rules.
- `test_anthropic_provider.py`, `test_openai_provider.py`, `test_gemini_provider.py`: request shapes, thinking modes, cache breakpoint placement, reasoning/signature replay, usage, error classification.
- `test_session.py`, `test_server.py`: coalescing, fan-out, attributed interrupts, interrupt-and-send, concurrent switches, presence, and two real WebSocket clients.
- `test_tools.py`: atomic edits, path containment, process-group kill, timeouts, stdin.
- `test_cli_pty.py`: the real CLI in a pseudo-terminal. Esc and Ctrl-C interrupt a running tool, and the next message isn't lost.

## 9. Demo script (≈8 min)

1. `uv run agent --workdir ./sandbox --model opus`: "create a python fizzbuzz in fizz.py, run it, and fix any bug". Shows thinking, tool calls, and `cache read=…` growing on each iteration.
2. While it's working, press **Esc** → the summary prints. Type "actually make it print in reverse". It continues cleanly.
3. `/switch codex` → "add a unit test and run it". OpenAI picks up the history Claude wrote. `/history` shows tool results, the partial message and the synthetic note.
4. "run `sleep 60`" → Esc mid-tool → "killed bash after …". The next message works.
5. `/switch haiku-4.5` right after an interrupted tool → the manual-thinking notice (rule 4 of the resume table).
6. `uv run agent-server --workdir ./sandbox`, two tabs (alice and bob): both type at once → queued + coalesced. bob presses Stop on alice's turn. The model dropdown switches for everyone.
