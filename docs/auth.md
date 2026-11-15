# API keys, limits and usage

Every `/api` and `/v1` call needs a key (`/api/health` doesn't). Issue one key per use case, such as Hearth,
Claude Code, the eval runner or a script, so each one gets its own limits, its own line in the usage report,
and can be revoked without touching the others.

## Concepts

| field | meaning |
|---|---|
| `owner` | The account, team or app the key belongs to. Conversations belong to the owner, so every key with the same owner sees the same saved chats, and rotating a key keeps them. |
| `name` | The use case: `hearth`, `claude-code`, `evals`, `nightly-script`. Shown in usage reports. |
| `scopes` | `chat` (the model, conversations, `/v1/*`) and/or `admin` (the `/api/admin` API). `admin` implies `chat`. |
| `models` | Allowlist of model ids (aliases are stored as their ids). Unset means every model in `models.toml`. Requests for other models get 403. |
| `rate_limit_per_minute` | Requests per minute. Unset means `KINDLING_RATE_LIMIT_PER_MINUTE` (default 60). Over the limit: 429 with `Retry-After`. |
| `daily_token_limit` | Prompt + output tokens per UTC day. Unset means unlimited. Over the limit: 429 until 00:00 UTC. |
| `expires_at` | Optional. After it, the key is rejected like an unknown one. |
| `note` | Free text: who has it, where it's configured. |

Keys look like `kdl_` plus 43 random characters (32 bytes). The full key is shown **once**, when it's created or
rotated. The server stores only its SHA-256, plus a 12-character `prefix` (`kdl_AbCdEfGh`) that lets you tell keys
apart in listings. A lost key can't be recovered; rotate it.

## Sending a key

Either header works, so OpenAI and Anthropic SDKs work unchanged:

```
Authorization: Bearer kdl_...
x-api-key: kdl_...
```

## First setup: the bootstrap key

There are no keys in a fresh database. Use either of these:

- **The CLI** (recommended). It talks to Postgres directly, so the server doesn't need to be running:
  `kindling db upgrade`, then `kindling keys create ...` as shown below.
- **The bootstrap key.** Set `KINDLING_API_KEY` to a long random value (`python -c "import secrets; print(secrets.token_urlsafe(32))"`),
  start the server, and use the admin API with it. It has `admin` + `chat` scopes, owner `admin`, and is rate
  limited at the default rate but has no token quota. It's compared in constant time.

Once you've issued real keys, including an admin key if you want to manage keys over HTTP, **remove
`KINDLING_API_KEY` and restart**. The bootstrap key can't be revoked, expired or scoped, and it doesn't show in
`keys list`. Keep it for CI and first setup only.

## Issuing keys per use case

```sh
# Hearth, the chat UI: generous rate limit, all models.
kindling keys create --owner gopal --name hearth --rpm 120 --note "Hearth on the Mac mini"

# Claude Code through the Anthropic-compatible API: one model, a daily budget, expires in 90 days.
kindling keys create --owner gopal --name claude-code --models nemotron --daily-tokens 2000000 --expires 90d

# The eval runner: its own owner, so its conversations and usage stay separate.
kindling keys create --owner evals --name runner --rpm 30 --daily-tokens 5000000

# A script: tight limits.
kindling keys create --owner gopal --name nightly-summary --rpm 5 --daily-tokens 200000 --expires 30d

# An admin key for managing keys over HTTP (instead of the bootstrap key).
kindling keys create --owner gopal --name admin --scopes admin,chat --expires 180d
```

Then configure each client:

- **Hearth:** set its API key setting to the `hearth` key.
- **Claude Code:** `ANTHROPIC_BASE_URL=https://<your host>` and `ANTHROPIC_API_KEY=kdl_...` (sent as `x-api-key`).
- **OpenAI SDK / scripts:** `OPENAI_BASE_URL=https://<your host>/v1`, `OPENAI_API_KEY=kdl_...`.
- **curl:** `curl -H "Authorization: Bearer $KINDLING_KEY" https://<your host>/api/info`.

`--expires` takes a duration (`30m`, `12h`, `90d`, `2w`) or a date (`2027-01-31`, UTC).

## Managing keys

```sh
kindling keys list [--owner gopal] [--all]      # --all includes revoked keys
kindling keys show kdl_AbCdEfGh                 # by prefix (or any unambiguous start of it), or by id
kindling keys rotate kdl_AbCdEfGh               # new secret, same owner/name/scopes/models/limits/expiry; the old one stops working
kindling keys revoke kdl_AbCdEfGh               # stops working; kept for the record
kindling usage --since 7d --by key              # or --by model, --by day; --owner, --key
```

The same over HTTP, with an `admin` key:

| method | path | |
|---|---|---|
| POST | `/api/admin/keys` | `{"owner", "name", "scopes", "models", "rate_limit_per_minute", "daily_token_limit", "expires_at" or "expires_in_days", "note"}`. Returns `key` (once) and the metadata. |
| GET | `/api/admin/keys` | `?owner=`, `?include_revoked=true` |
| GET | `/api/admin/keys/{id}` | metadata plus `today`: requests and tokens so far (UTC) |
| PATCH | `/api/admin/keys/{id}` | any of `name`, `note`, `scopes`, `models`, `rate_limit_per_minute`, `daily_token_limit`, `expires_at`. Only the fields you send change; `null` clears models, limits or expiry. |
| POST | `/api/admin/keys/{id}/revoke` | |
| POST | `/api/admin/keys/{id}/rotate` | returns the new `key` once, and `replaces`: the old id |
| GET | `/api/admin/usage` | `?owner=&key_id=&since=7d&until=&group_by=key\|model\|day`, which returns requests, errors, prompt/output tokens and wall seconds |

Validation: `owner` and `name` are 1 to 100 characters of letters, digits and `. _ @ : -`, starting with a letter or
digit. Scopes must be a subset of `chat` and `admin`. Models must exist in `models.toml`; a 422 lists the available ones.

**Rotation** creates a new key with a new id and revokes the old one in one transaction. Switch the client to the
new key straight away: the old one stops working at once. The owner is unchanged, so saved conversations carry over.
The new key starts a fresh rate-limit bucket and daily token count.

**Revocation** through the admin API takes effect immediately. Through the CLI, a running server picks it up within
about 10 seconds, the lifetime of its key lookup cache.

## Limits: how they're counted

- **Rate limit:** a token bucket per key that allows `rate_limit_per_minute` requests per minute, in bursts of up to
  that many. It applies to every authenticated request: reads, chats and admin calls.
- **Daily token quota:** prompt + output tokens per UTC day, checked *before* the model is called. A request that
  starts under the limit runs to completion, so a key can end the day slightly over. Only requests that run the
  model are refused: non-GET requests outside `/api/admin` and `/api/conversations`. Reads and conversation
  management keep working when a key is over quota.
- Both are **per process**, kept in memory. The token counter is seeded from `usage_events` the first time a key is
  checked each day (so a restart doesn't reset it), then kept current by the usage recorder without querying the
  database. Kindling runs as a single process. If you ever run several workers, each one enforces its own rate
  limit, and the quotas become approximate.
- `last_used_at` is written at most about once a minute per key.
- With `KINDLING_AUTH_DISABLED=true` there are no keys and no limits.

## Usage metering

Every model request writes one row to `usage_events`: owner, key id and name, endpoint, model, prompt and output
tokens, wall time, and status (`ok`, `error`, or `stopped` when the client hung up). Metering never fails a request:
if the write fails, it's logged and the chat carries on. The history outlives the keys, because there's no foreign key.

## Security notes

- **Keys are secrets.** Anyone holding one can use the model as that owner and read that owner's saved
  conversations. Give each use case its own key, the narrowest scopes and model list it needs, an expiry, and a
  token budget.
- **Only hashes are stored.** A database dump doesn't yield working keys. Keys never appear in logs: rejected
  requests are logged at debug level with a reason (unknown, revoked, expired) and, for known keys, the key id.
- **Every bad key gets the same 401** (`missing or invalid API key`, `WWW-Authenticate: Bearer`), whether it's
  unknown, revoked or expired. Keys are 256-bit random values, so guessing one isn't feasible. Use Cloudflare
  (WAF / rate rules) to absorb floods of bad requests before they reach the Mac.
- **The bootstrap key** is the most powerful credential and can't be revoked. Remove it from the environment
  after setup.
- **Admin keys** can mint more keys. Keep them few, give them expiries, and don't put them in client apps.
- **Behind Cloudflare Tunnel** the server is reachable from the internet even though it listens on 127.0.0.1.
  Never set `KINDLING_AUTH_DISABLED=true` there.
- If the key database is unreachable, requests fail closed with 503, not open.
