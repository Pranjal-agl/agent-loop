# agent-loop (scaffold)

This is a design sketch, not a finished submission. It exists to
sanity-check the architecture before the real implementation work starts.

## What's here and actually runnable
- Full folder structure and package layout
- `core/state.py` — provider-agnostic `ConversationState`, content blocks,
  and an explicit `LoopState` state machine with a transition table
- `providers/base.py` — the `AgentProvider` protocol every provider
  implements, plus the `StreamEvent` union
- `providers/mock_provider.py` — a working zero-cost fake provider
- `providers/anthropic_provider.py` — real Claude adapter: extended
  thinking, explicit prompt-cache breakpoints, and the synthetic-
  tool_result resume invariant. Verified via a mocked-transport test
  suite (`tests/test_anthropic_provider.py`) that fakes the SDK's
  streaming shape — **not yet exercised against a live API call**, since
  no `ANTHROPIC_API_KEY` was available while building this. Smoke-test
  against the real API before the demo.
- `providers/openai_provider.py` — real OpenAI adapter using the
  Responses API (chosen over Chat Completions specifically for reasoning-
  model support — see the module docstring for why). Same caveat: verified
  via `tests/test_openai_provider.py` against a faked transport, not a
  live call.
- `tools/` — working `file_read`, `file_edit`, and `bash` (cancellation-
  safe subprocess pattern + a best-effort destructive-command denylist)
- `core/loop.py` — the orchestrator, including the queue-based bridge
  that makes mid-stream interruption actually cancellable
- `core/interrupt.py` — the interrupt manager
- `cli.py` — REPL with `/switch <provider> <model>`, and a real interrupt
  key-watcher (Ctrl-C or Esc, raw-terminal level, independent of whether
  a prompt is active) — **needs a real TTY to test properly**; this
  sandbox has none, so the watcher's no-TTY fallback path is what's been
  exercised here, not an actual keypress.
- `tests/` — 17 passing tests across the state machine, both provider
  adapters (mocked transport), and the bash denylist

Run it now with zero API keys:
```
uv sync
uv run pytest
uv run agent   # mock provider only until you export API keys
```

Add real providers by exporting keys before running:
```
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
uv run agent
> /switch anthropic claude-haiku-4-5
```

## Explicitly NOT built yet (next steps)
- Live-API smoke test for both real adapters (blocked on API keys)
- Real-terminal test of the Ctrl-C/Esc interrupt watcher (blocked on TTY)
- Web UI (FastAPI + SSE/WebSocket), reusing `AgentLoop` as-is
- Multiplayer mode design + optional toy implementation

## Key design decisions baked into this scaffold (open to revisiting)
1. **Interrupt semantics**: an in-flight tool call is allowed to finish
   rather than hard-killed, but its result is reported back to the model
   as cancelled. Rationale: killing a `bash` command mid-write risks a
   half-written file being *worse* than either fully-done or
   fully-reverted. `BashTool` does kill the *subprocess* on cancellation
   (can't leave orphaned processes) but the design still treats the
   *result* as void either way.
2. **State machine is data** (`VALID_TRANSITIONS` dict), not scattered
   conditionals — makes it possible to unit test "what are all reachable
   states" independent of any provider.
3. **ConversationState is provider-agnostic and pydantic-typed**, so
   `switch_provider()` mid-conversation is just changing two string
   fields — no conversion happens until the next `stream_turn()` call.
4. **Partial output is never silently dropped.** Text that streamed
   before an interrupt is captured into a real `Message` with
   `partial=True` and appended to history, rather than discarded — so
   the next turn's context (and the transcript the user sees) reflects
   what was actually shown, not a gap.

## bugs to be fixed
Running the example tests during scaffolding surfaced two real bugs in
the first draft, both instructive for the interrupt-handling writeup:
- Cancelling the provider task through the queue-bridge in
  `_stream_with_cancellation` initially got translated into a normal
  "stream ended" signal, so the loop happily continued to `DONE`
  instead of `INTERRUPTED` — the cancellation was real, but it never
  reached the state machine. Fixed by distinguishing a `_CANCELLED`
  queue marker from the normal `_SENTINEL`.
- The assistant's completed message was never being appended to
  `state.messages` at all (only yielded as events) — caught by
  `test_switch_provider_mid_conversation` asserting on message count.
- (Found later, in `cli.py`) The interrupt-key watcher's "no real TTY"
  fallback originally just `return`ed immediately. Under piped/non-TTY
  input, `asyncio.wait(FIRST_COMPLETED)` then saw the watcher finish
  *before* the turn did, on every single turn, and misread that as "the
  user pressed the interrupt key" — so every turn appeared interrupted
  immediately, even though nothing was ever pressed. A CLI smoke test
  with piped input caught this in about ten seconds; it would have been
  much harder to notice in an interactive terminal session, since a real
  developer typically wouldn't be leaving stdin unconnected. Fixed by
  having the no-TTY branch await the same event forever instead of
  returning, so it only ever ends via explicit cancellation.
Both are the kind of bug that's invisible until you actually exercise
interruption in a test, which is exactly why `MockProvider` +
`pytest-asyncio` exist in this scaffold from day one instead of being
bolted on later.

## Multiplayer mode (design)

Not implemented — this is the design the spec asks for even without code.
Grounded in what already exists: a `ConversationState` + `AgentLoop` pair
is currently a single-owner, single-process object. Multiplayer means
letting more than one client attach to the *same* pair.

**Session model.** A session is one `(ConversationState, AgentLoop)`,
keyed by a session id, held server-side (a `dict[str, AgentLoop]` is
enough to start; Redis/Postgres later per the storage-interface note
above). Any client — CLI or web — that knows the session id can attach.
A thin FastAPI + WebSocket layer would own this dict; each connected
client gets a fan-out `asyncio.Queue` fed from the single loop's event
stream, so everyone watching a session sees identical `StreamEvent`s in
real time, not a per-client re-run.

**Two users sending a message "at the same time."** The loop's state
machine only accepts a new user message from `IDLE` or `DONE`
(`VALID_TRANSITIONS`), so at most one message can actually start a turn.
The rest is a queuing question, not a data-race question: incoming
messages are appended to a per-session FIFO in server-receive order, and
the loop drains one at a time. Whoever's message the server saw first
goes first — no fancier resolution needed, since this is exactly how a
shared group chat already behaves and reinventing that would surprise
people.

**A message arriving *while a turn is running*.** This is the more
interesting case, and the one worth being explicit about: should a
second user's message barge in (auto-interrupt the first user's
in-flight turn) or queue behind it? Default here is **queue, don't
auto-interrupt** — silently cutting off someone else's request because
a second person happened to type first is a worse failure mode than
making them wait a few seconds. Interrupting an in-flight turn is still
possible, but as an explicit action ("Stop" button / `/interrupt`
command) any participant can take, distinct from just sending a new
message. That action calls the existing `AgentLoop.handle_interrupt()`
unchanged — multiplayer doesn't need a new interrupt mechanism, only a
policy for who's allowed to trigger the existing one and how it's
surfaced to everyone else in the room (see "attribution" below).

**Real-time visibility.** Yes — every `StreamEvent` (text/thinking
deltas, tool call start/end, `MessageComplete`) is broadcast to all
attached clients as it's produced, since they're already reading off the
same fan-out queues. Presence ("Alex is typing...") is treated as a
separate, ephemeral side-channel — ping-style events that never touch
`ConversationState` — since presence is inherently lossy/best-effort and
polluting the durable conversation history with it would be a mistake.

**Attribution.** `Message` doesn't currently carry a sender identity —
worth adding a `author_id: str | None` field before multiplayer, so the
UI can show "Priya interrupted the last turn" or "Sam switched to
Opus" rather than an anonymous system action. Small, additive change to
the existing pydantic model; the state machine and provider adapters are
untouched by it.

**Conflicts elsewhere in state.** `InterruptManager.interrupt()` is
already idempotent — a second concurrent interrupt call on an
already-cancelled task just returns `False` — so two users hitting
"Stop" at once doesn't double-transition the state machine or raise.
`switch_provider()` is not currently guarded, though: two simultaneous
`/switch` calls would race on plain attribute writes. That's a real gap
for a full implementation — wrapping `ConversationState` mutations
outside of `run_turn` (provider switches, config changes) in a per-session
`asyncio.Lock` closes it without touching the core loop logic.

**What a minimal toy implementation would look like**, if time allowed
past everything above: a FastAPI process holding the session dict, one
WebSocket route that both accepts `{"type": "user_message", "text": ...}`
/ `{"type": "interrupt"}` payloads and streams `StreamEvent`s back
JSON-encoded, and two browser tabs (or two CLI instances pointed at the
same session id) proving the fan-out actually works. That's a few hours
of additional work on top of this scaffold, not a redesign.

| | Claude (Anthropic) | OpenAI (Responses API) |
|---|---|---|
| Reasoning exposure | Full thinking blocks, replayable via `signature` | Reasoning *summary* only; raw CoT not exposed |
| Prompt caching | Explicit `cache_control` markers per content block | Automatic, keyed by prompt prefix — no manual breakpoints |
| Resume after interrupt | Dangling `tool_use` **must** get a matching `tool_result` or the API rejects the request — scaffold synthesizes a "cancelled" one | Dangling `function_call` can simply be omitted from the next `input` — scaffold drops it |

This table is the concrete answer to "each provider has different
invariants for resuming from an interrupted state" from the spec — see
`_to_anthropic_messages` vs `_to_input_items` for the code, and
`tests/test_anthropic_provider.py::test_dangling_tool_use_gets_synthetic_result_before_next_call`
/ `tests/test_openai_provider.py::test_dangling_function_call_is_dropped_not_synthesized`
for the tests that pin this behavior down.
