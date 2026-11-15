# Kindling

A private AI assistant backend. Kindling serves **NVIDIA Nemotron** (or any Ollama or OpenAI-compatible model) to apps, scripts and people: streaming chat with tool calling, saved conversations in Postgres, API keys per use case, OpenAI- and Anthropic-compatible endpoints, and an eval harness. It runs on your own machine; [Hearth](../hearth) is its chat UI.

```
Hearth · OpenAI / Anthropic SDKs · Claude Code · scripts
        │  Authorization: Bearer kdl_…
        ▼
Kindling (FastAPI) ── keys, limits, usage ── Postgres + pgvector
        │  models.toml: "nemotron" → nemotron-3-nano:30b
        ▼
Ollama (this Mac)  ·  vLLM / NIM (a GPU box, later)
```

## Quick start

You need [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) with `ollama pull nemotron-3-nano:30b`, and Docker for Postgres.

```bash
uv sync
docker compose up -d db                   # Postgres + pgvector on localhost:5433
uv run kindling db upgrade                # create the tables
cp models.example.toml models.toml        # the models this machine offers
uv run kindling keys create --owner you --name hearth --scopes chat   # shown once: keep it
uv run kindling serve                     # http://127.0.0.1:8000, docs at /docs
```

Every call needs a key (`Authorization: Bearer kdl_…` or `x-api-key`). For Hearth, put it in Hearth's `.env.local` as `CHAT_API_KEY`; Hearth's dev server adds it to each request, so the browser never sees it. `uv run kindling check` tells you whether the model, the database and the models file are in order.

## What's inside

| Area | What you get | Docs |
|---|---|---|
| Chat | `POST /api/chat`: newline-delimited JSON events (reasoning, answer, tool calls, stats), keepalive pings for proxies, optional saving | below |
| Tools | the model can call `calculate`, `current_time`, `list_files`, `read_file` (only inside `workspace/`); add your own in `src/kindling/tools/` | `tools/registry.py` |
| Conversations | `/api/conversations`: list, read, rename, delete; each key's owner sees only their own | |
| Models | a per-machine `models.toml`: stable ids and aliases (`nemotron`, `nemotron-fast`) over Ollama or vLLM / NIM | [models.example.toml](models.example.toml) |
| API keys | one key per use case, with scopes, model allowlists, requests/minute, daily token quotas, expiry, rotation, usage reports | [docs/auth.md](docs/auth.md) |
| Standard APIs | `/v1/chat/completions`, `/v1/models` (OpenAI), `/v1/messages` (Anthropic: SDKs and Claude Code) | [docs/api.md](docs/api.md) |
| Operations | JSON logs with request ids, CORS, size limits, a queue in front of the model, Prometheus `/metrics`, Docker, CI | [docs/deploy.md](docs/deploy.md) |
| Cloudflare | Tunnel and AI Gateway setup scripts, origin protection, client IPs | [docs/cloudflare.md](docs/cloudflare.md) |
| Evals | `kindling eval run smoke`: datasets, scorers, runs in-process, over HTTP or against any OpenAI server; compare runs | [docs/evals.md](docs/evals.md) |

## Keys for different use cases

```bash
uv run kindling keys create --owner you --name claude-code --models nemotron --rpm 30 --daily-tokens 2000000
uv run kindling keys create --owner you --name evals --expires 30d
uv run kindling keys list
uv run kindling usage --by key            # who used how much
uv run kindling keys rotate <id|prefix>   # new secret, same owner and history
uv run kindling keys revoke <id|prefix>
```

The same is available over HTTP under `/api/admin` with an admin-scoped key.

## Another machine, a bigger Nemotron

Each machine keeps its own `models.toml` (it's gitignored). Keep the ids stable and change what's underneath, so clients and keys don't change: on a 128 GB machine point `nemotron` at `nemotron-3-super:120b`, or at a vLLM / NIM server with `provider = "openai"`. Then compare the two with the eval harness:

```bash
uv run kindling eval run smoke --label nano
uv run kindling eval run smoke --label super      # on the other machine
uv run kindling eval compare evals/results/*-nano.jsonl evals/results/*-super.jsonl
```

## Development

```bash
uv run pytest -q                          # offline: a scripted model and an in-memory store
KINDLING_TEST_DATABASE_URL=postgresql+asyncpg://kindling:kindling@localhost:5433/kindling_test uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

Settings are environment variables (`KINDLING_…`) or a `.env` file; [.env.example](.env.example) lists them all. Layout: `src/kindling/` (`api/` routes, `llm/` models, `agent.py` the tool loop, `tools/`, `store/` Postgres and memory stores, `auth.py` and `keys.py`, `usage.py`, `ops/`, `edge.py`, `evals/`), `migrations/` (Alembic), `deploy/`, `docs/`.

## Roadmap

Long-term memory and retrieval on pgvector, more tools (web, SQL, MCP), harness and agent-trajectory evals, conversation history in Hearth.

## License

MIT
