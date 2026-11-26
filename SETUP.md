# Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Zero-cost path (no keys)
```
uv sync --all-extras
uv run pytest                      # 86 tests, no network calls
uv run agent --workdir ./sandbox   # CLI on the mock provider
```
The mock provider is enough to demo the loop mechanics: `!<command>` makes it call `bash`. For example, run `!echo hi; sleep 30`, then press Esc to interrupt mid-tool.

## Real providers
Put keys in the environment or in a `.env` file in the directory you run from:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...          # or GOOGLE_API_KEY
AGENT_PROVIDER=anthropic    # optional default provider
AGENT_MODEL=claude-opus-5-5 # optional default model
AGENT_MAX_ITERATIONS=50     # optional
```
Only providers with keys are registered. Model aliases (`/switch opus`, `codex`, `gemini`, …) live in `src/agent_loop/config.py`. Edit them if model names change. Any full model id also works: `/switch openai gpt-5.5`.

## Live smoke test (a few cents)
```
uv run python scripts/smoke_test_live.py                       # every configured provider
uv run python scripts/smoke_test_live.py anthropic --model anthropic=claude-haiku-4-5
```
For each provider it runs: a tool loop, an interrupt mid-stream and a continuation, an interrupt mid-tool and a continuation, and a prompt-caching check. With 2+ providers it also runs a cross-provider conversation. Run this before the demo.

## Multiplayer server
```
uv run agent-server --workdir ./sandbox --port 8000
open "http://127.0.0.1:8000/?session=demo&user=alice"
open "http://127.0.0.1:8000/?session=demo&user=bob"
```

## Interrupt keys
In the CLI, **Esc** or **Ctrl-C** while the agent is working interrupts it. This needs a real terminal. With piped stdin, send SIGINT instead (`kill -INT <pid>`). Ctrl-C at an empty prompt clears the line, and Ctrl-D quits.
