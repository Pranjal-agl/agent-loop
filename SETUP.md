# Setup

## Zero-cost path (default, no keys needed)
```
uv sync
uv run pytest        # 17 tests, no network calls, no cost
uv run agent          # CLI against the mock provider
```

## Adding real providers
```
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
uv run agent
> /switch anthropic claude-haiku-4-5
```
Either key can be set independently — the CLI only registers the
providers it has keys for; `mock` is always available regardless.

Optional overrides (see `config.py`):
```
export AGENT_PROVIDER=anthropic   # default provider on startup
export AGENT_MODEL=claude-haiku-4-5
```

## Live smoke test (costs a small amount of real money)
The pytest suite is entirely mocked-transport — no network calls, no
cost, safe to run anytime. To confirm the real adapters actually work
against the live APIs (not yet verified as of this scaffold — see
README), run:
```
uv run python scripts/smoke_test_live.py            # tests whichever keys are set
uv run python scripts/smoke_test_live.py anthropic   # just one
uv run python scripts/smoke_test_live.py openai
```
Each call uses the cheapest current small model and a one-word response,
so the cost per run is a small fraction of a cent per provider tested.

## Testing the interrupt key-watcher
`cli.py`'s Ctrl-C/Esc interrupt watcher needs a real terminal (TTY) to
do anything — it safely no-ops under piped input (used for the CLI smoke
tests during development), but that means an actual keypress has never
been exercised. Run `uv run agent` in a real terminal, send a message,
and press Ctrl-C or Esc while it's streaming to confirm the interrupt
actually fires before relying on it for the demo.
