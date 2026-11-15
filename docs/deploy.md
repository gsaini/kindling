# Running Kindling as a service

Kindling is one process: a FastAPI app that talks to Ollama for the model and to Postgres for
conversations, keys and usage. Ollama always runs on the Mac itself, because Docker on macOS has no
GPU / Metal access. Kindling and Postgres can run natively or in Docker.

To reach it from outside this machine, put Cloudflare in front: see [cloudflare.md](cloudflare.md).

## Configuration

Every setting is an environment variable `KINDLING_<NAME>`, read from the environment or from a
`.env` file in the working directory. [`.env.example`](../.env.example) lists all of them with their
defaults and a line on each:

```sh
cp .env.example .env
cp models.example.toml models.toml   # the models this machine offers (optional but recommended)
```

Then set at least `KINDLING_API_KEY` (generate one with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`).

## Local development (uv)

```sh
uv sync
docker compose up -d db          # Postgres + pgvector on 127.0.0.1:5433
uv run kindling db upgrade       # apply migrations
uv run kindling check            # model reachable? database up? models file valid?
uv run kindling serve            # http://127.0.0.1:8000, docs at /docs
```

No database at hand? `KINDLING_STORE=memory uv run kindling serve` keeps everything in memory.

Tests run offline with a scripted fake model: `uv run pytest -q`. The Postgres tests run too when
`KINDLING_TEST_DATABASE_URL=postgresql+asyncpg://kindling:kindling@localhost:5433/kindling_test` is
set (the `db` service creates that database on first start).

## Docker Compose

The database only (what development uses):

```sh
docker compose up -d db
```

The database and the API server, both in containers. The `app` service sits behind the `app`
profile so the command above never starts it:

```sh
cp models.example.toml models.toml    # first, or Docker creates an empty models.toml directory
docker compose --profile app up -d --build
docker compose --profile app logs -f app
curl http://127.0.0.1:8000/livez
```

What the app service does:

- reads `.env`, then overrides the hosts for life in a container: the database is `db:5432`, Ollama
  is `http://host.docker.internal:11434` (the Mac), and the server listens on `0.0.0.0` inside the
  container. The port is published on `127.0.0.1:8000` only.
- mounts `./models.toml` and `./workspace` read-only.
- runs migrations on every start (when `KINDLING_STORE=postgres`), then `kindling serve`. Migrations
  are idempotent, so restarts are safe.
- runs as an unprivileged user, and Docker checks `GET /livez` every 30 seconds.

Ollama must listen where the container can reach it. The Ollama app listens on 127.0.0.1 only, which
Docker Desktop's `host.docker.internal` still reaches. If it doesn't, set `OLLAMA_HOST=0.0.0.0` for
Ollama (see below) and keep the Mac's firewall on.

The image builds on its own too: `docker build -t kindling .`. It contains the project at `/app`
(code, `migrations/`, `alembic.ini`, `models.example.toml`) and a virtualenv built by
`uv sync --frozen --no-dev`.

## Health, metrics and logs

| Endpoint | Purpose | Auth |
| --- | --- | --- |
| `GET /livez` | Is the process up? No dependencies; for container and process health checks. | none |
| `GET /api/health` | Is it ready? Checks the default model and the database. | none |
| `GET /metrics` | Prometheus metrics. Turn off with `KINDLING_METRICS_ENABLED=false`. | admin key, unless auth is disabled |

Metrics include requests by method, route template and status, a latency histogram (to the end of
the stream for streaming replies), generations in flight, the queue depth, queue waits and
rejections, and prompt / output tokens by model. Give Prometheus its own key
(`kindling keys create --owner ops --name prometheus --scopes admin`) and a scrape job:

```yaml
scrape_configs:
  - job_name: kindling
    authorization:
      credentials_file: /etc/prometheus/kindling-key   # a key with the admin scope
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

Logs go to stderr, one access line per request plus application messages. `KINDLING_LOG_FORMAT=json`
writes JSON lines (`ts`, `level`, `logger`, `msg`, `request_id`, `method`, `path`, `status`,
`duration_ms`, `client_ip`, `key_owner`, `key_name`) for a log shipper. Logs never contain request
bodies, prompts, replies, query strings, or the `Authorization` / `x-api-key` headers.

Every response carries `X-Request-ID`. Send your own (letters, digits, `.`, `_`, `:`, `-`; up to 128
characters) to follow a request through your logs and Kindling's; otherwise Kindling makes one.

## Limits and overload

- **Request size**: bodies over `KINDLING_MAX_REQUEST_BYTES` (2 MB) get 413, chunked bodies included.
- **Concurrency gate**: every model stream, across every model and API, goes through one gate. At most
  `KINDLING_MAX_CONCURRENT_GENERATIONS` run at once; up to `KINDLING_MAX_QUEUED_GENERATIONS` wait,
  each for at most `KINDLING_QUEUE_TIMEOUT_SECONDS`. Anything beyond that is turned away at once with
  503 (529 on the Anthropic-compatible API) and `Retry-After`, which clients and SDKs retry. A slot is
  freed when the model finishes or the client disconnects.
- **Rate limits** are per key (`KINDLING_RATE_LIMIT_PER_MINUTE`, overridable per key).

## Tuning Ollama

Set Ollama's variables for the app with `launchctl setenv NAME value` and restart Ollama, or run
`NAME=value ollama serve` yourself.

- **`OLLAMA_NUM_PARALLEL` versus `KINDLING_MAX_CONCURRENT_GENERATIONS`.** Ollama serves this many
  requests per loaded model at once and queues the rest internally, without telling anyone. Set
  Kindling's limit to the same number, so the queue lives in Kindling instead: bounded, visible in
  `/metrics`, and answered with a quick 503 when full. Each parallel slot reserves its own context
  window of memory, so raising it costs memory as well as per-request speed; on one Mac, 1 or 2 is
  usually right for a 30B model.
- **`OLLAMA_CONTEXT_LENGTH` versus `num_ctx`.** Kindling asks for each model's `num_ctx` (from
  `models.toml`, or `KINDLING_NUM_CTX`) on every request. When a request asks for a different window
  than the one the model was loaded with, Ollama reloads the model, which takes many seconds for a
  30B model. Set `OLLAMA_CONTEXT_LENGTH` to the same value, so other clients of the same Ollama
  (`ollama run`, editors, scripts) load the model with the same window and don't trigger reloads.
- **`OLLAMA_KEEP_ALIVE`.** How long a model stays loaded after its last request (default 5 minutes).
  For a server, use something long (`24h`, or `-1` for forever), so the first request after a quiet
  spell doesn't pay the load time. The memory stays in use while it's loaded.
- If `models.toml` offers two Ollama models (say `nemotron` and `nemotron-fast`), make sure
  `OLLAMA_MAX_LOADED_MODELS` and the machine's memory allow both to stay loaded, or every switch
  between them is a reload.

## Production checklist

- [ ] **Auth on**: `KINDLING_AUTH_DISABLED=false` (the default). A tunnel to a 127.0.0.1 server is
      reachable from the internet.
- [ ] **One key per use case**: `kindling keys create --owner <team> --name <use case>` for each app,
      tool or person, with only the scopes and models it needs. Revoke or rotate one without
      touching the others.
- [ ] **Remove the bootstrap key** after setup: delete `KINDLING_API_KEY` from `.env` and restart.
- [ ] **Keep the port on loopback** (`KINDLING_HOST=127.0.0.1`, or the compose default
      `127.0.0.1:8000`) and come in through Cloudflare ([cloudflare.md](cloudflare.md)), with
      `KINDLING_ORIGIN_SECRET` set so only the edge can call the origin.
- [ ] **CORS** only for browser apps that call the API directly, listing their exact origins.
- [ ] **Back up Postgres**. The data lives in the `kindling-db` volume. A nightly dump:
      `docker compose exec -T db pg_dump -U kindling -Fc kindling > kindling-$(date +%F).dump`,
      restored with `pg_restore -U kindling -d kindling --clean`. Test a restore once.
- [ ] **Ship logs**: `KINDLING_LOG_FORMAT=json` and collect stderr (Docker's log driver, or the
      launchd / systemd journal for a native install).
- [ ] **Scrape metrics** with an admin-scoped key, and alert on 5xx rates, a full queue
      (`kindling_generations_rejected_total`) and `/livez` failures.
- [ ] **Ollama tuned** as above: parallelism matched, context lengths matched, keep-alive long.
