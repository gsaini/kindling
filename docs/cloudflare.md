# Kindling behind Cloudflare (Tunnel and AI Gateway)

Kindling listens on `127.0.0.1:8000` and nothing outside the Mac can reach it. This guide puts it on a
public hostname with **Cloudflare Tunnel**, adds an optional **AI Gateway** lane for SDKs and automation,
and covers keys, clients, limits and troubleshooting. The scripts are in [deploy/cloudflare](../deploy/cloudflare/).

Research for this guide was done against developers.cloudflare.com in November 2026. Things Cloudflare's
docs don't state, and which you should check with the curl commands at the end, are marked **(unverified)**.

## Contents

1. [Topology](#topology)
2. [What you need to provide](#what-you-need-to-provide)
3. [Keys: which goes where](#keys-which-goes-where)
4. [Setup, step by step](#setup-step-by-step)
5. [Client configuration](#client-configuration)
6. [Origin protection and client IPs (Kindling settings)](#origin-protection-and-client-ips)
7. [Gateway features: use, avoid, costs](#gateway-features-use-avoid-costs)
8. [Timeouts and long responses](#timeouts-and-long-responses)
9. [Troubleshooting: check each hop](#troubleshooting-check-each-hop)
10. [What was verified, with sources](#what-was-verified-with-sources)

## Topology

```
                         Authorization: Bearer kdl_...     (Kindling key, always)
                         cf-aig-authorization: Bearer ...  (gateway token, gateway lane only)

  OpenAI / Anthropic SDKs ─┐
  automation (n8n, cron) ──┼─> AI Gateway  gateway.ai.cloudflare.com/v1/<acct>/<gw>/custom-kindling/...
                           │      logs/analytics, optional cache, retries     │
                           │                                                  │ base_url + path
  Hearth (browser) ────────┤                                                  v
  Claude Code ─────────────┼──────────────────────────────> https://kindling.example.com   (Cloudflare edge)
  other people's tools ────┘                                  [optional: Access]
                                                              [Transform Rule adds X-Kindling-Origin-Secret]
                                                                        │
                                                                        │ Cloudflare Tunnel (outbound from the Mac)
                                                                        v
                                                              cloudflared on the Mac (launchd)
                                                                        │
                                                                        v
                                                              Kindling  http://127.0.0.1:8000
                                                              (keys, scopes, per-key rate limits,
                                                               edge middleware, usage)
                                                                        │
                                                                        v
                                                              Ollama    http://127.0.0.1:11434
                                                              (nemotron-3-nano)
```

Two lanes reach the same hostname:

- **Direct lane** (`https://kindling.example.com`): the default for everything. Hearth, Claude Code and
  anyone you hand a Kindling key to. Fewer moving parts, no Cloudflare token to distribute, nothing stored
  at Cloudflare.
- **Gateway lane** (`https://gateway.ai.cloudflare.com/v1/<account>/<gateway>/custom-kindling/...`): for
  your own SDK scripts and automation, when you want Cloudflare's request analytics and logs, or gateway
  retries. AI Gateway reaches Kindling as a **custom provider** whose `base_url` is the tunnel hostname.

Why Hearth and Claude Code go direct:

- **Hearth** runs in a browser. The gateway lane needs a Cloudflare token in every request, and a token in
  browser code is a token anyone can read. The gateway also adds nothing Hearth needs.
- **Claude Code** speaks the Anthropic Messages API. Custom providers forward any path in the provider's
  own format, so `custom-kindling/v1/messages` should reach Kindling's `/v1/messages`, but Cloudflare only
  documents OpenAI-shaped examples for custom providers **(unverified for Anthropic format)**. Long agent
  sessions also stream for minutes, which is simpler with one less proxy. Going direct is the safe default;
  the gateway variant is below if you want to try it.
- **Other people** should only ever get a Kindling key. A gateway token (`AI Gateway: Run`) works on every
  gateway in the Cloudflare account, so it is not something to hand out.

## What you need to provide

- **A Cloudflare account you control.** Not the organisation account whose credentials are in
  `~/.cloudflared` on this Mac. Note its **Account ID** (dashboard, account home, right-hand column).
- **A domain on that account** (a zone with Cloudflare nameservers; Free plan is fine) and the hostname to
  use, e.g. `kindling.example.com`.
- **An API token for setup** (My Profile > API Tokens > Create Token > Custom):
  - Account > Cloudflare Tunnel > Edit
  - Account > AI Gateway > Edit
  - Zone > DNS > Edit, for that zone
  - Zone > Zone > Read, for that zone (or pass `CF_ZONE_ID` instead)
  - Zone > Transform Rules > Edit, for that zone (only if you use an origin secret)
  Delete or roll it once setup is done; nothing needs it at run time.
- **A gateway token for clients** (only for the gateway lane): created in the AI Gateway dashboard or as an
  API token with **AI Gateway > Run**. Cloudflare shows it once.
- **Kindling keys**, one per client or person (`kindling keys create ...`). These never go to Cloudflare's
  settings; they travel in each request.
- Optionally an **origin secret**: `openssl rand -hex 32`.

## Keys: which goes where

| Key | Where it lives | Who sends it | Header |
|---|---|---|---|
| Kindling key `kdl_...` | the client | every client, both lanes | `Authorization: Bearer kdl_...` or `x-api-key: kdl_...` |
| Gateway token | the client (gateway lane only) | your scripts, automation | `cf-aig-authorization: Bearer <token>` |
| Origin secret | Cloudflare Transform Rule + Kindling `.env` | Cloudflare's edge, never a client | `X-Kindling-Origin-Secret` |
| Setup API token | your shell, during `setup.sh --apply` | `setup.sh` | `Authorization: Bearer` to api.cloudflare.com |
| Access service token (optional) | the client | machine clients, if Access is on | `CF-Access-Client-Id` / `CF-Access-Client-Secret` |

**Recommendation: clients pass their own Kindling keys through the gateway.** For a custom provider the
gateway forwards `Authorization: Bearer ...` to the upstream as the provider key (documented), and its own
token travels separately in `cf-aig-authorization`. So Kindling still sees who is calling, applies that
key's scopes and rate limit, and records usage per key. Revoking a key in Kindling cuts that client off on
both lanes.

The alternative, storing one Kindling key in the gateway as the provider key (BYOK, Secrets Store), is
**not recommended**:

- Every gateway caller becomes the same Kindling key, so per-key limits, scopes and usage are lost.
- Whoever holds any `AI Gateway: Run` token for the account can then use Kindling, and those tokens can't
  be limited to one gateway.
- Cloudflare's BYOK docs don't say whether stored keys work for custom providers **(unverified)**, and if a
  client sends its own `Authorization` header the gateway forwards that instead of the stored key.

The one thing BYOK would buy (clients holding only a Cloudflare token) isn't worth losing identity in
Kindling.

## Setup, step by step

`setup.sh` does steps 1 to 3 and 5 to 6 through the Cloudflare API. Run it without `--apply` first: it
prints every request and changes nothing.

```bash
export CF_ACCOUNT_ID=<account id> KINDLING_HOSTNAME=kindling.example.com GATEWAY_ID=kindling
./deploy/cloudflare/setup.sh                     # dry run: read the plan
export CF_API_TOKEN=<setup token>
./deploy/cloudflare/setup.sh --apply             # do it
# with an origin secret too:
ORIGIN_SECRET=$(openssl rand -hex 32) ./deploy/cloudflare/setup.sh --apply
```

### 1. Tunnel

The script creates a **locally managed** named tunnel `kindling` through the API (`POST
/accounts/{id}/cfd_tunnel` with `config_src: "local"` and a generated `tunnel_secret`) and writes its
credentials file to `~/.kindling/cloudflared/<tunnel-id>.json`. This is the API equivalent of
`cloudflared tunnel create kindling`, without `cloudflared tunnel login` (which would write a `cert.pem`
into `~/.cloudflared`, where another account's credentials already live).

Dashboard alternative: Zero Trust > Networks > Tunnels > Create a tunnel (that makes a remotely managed
tunnel, run with a token; then ingress is set in the dashboard, not in `config.yml`).

### 2. Config and service

`~/.kindling/cloudflared/config.yml` is rendered from
[config.yml.template](../deploy/cloudflare/config.yml.template):

```yaml
tunnel: <tunnel-id>
credentials-file: /Users/you/.kindling/cloudflared/<tunnel-id>.json
ingress:
  - hostname: kindling.example.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Check it, then run it in the foreground once:

```bash
cloudflared tunnel --config ~/.kindling/cloudflared/config.yml ingress validate
cloudflared tunnel --config ~/.kindling/cloudflared/config.yml ingress rule https://kindling.example.com
cloudflared tunnel --config ~/.kindling/cloudflared/config.yml run
```

Always pass `--config`: without it cloudflared reads `~/.cloudflared/config.yml`.

To run at login, use the rendered LaunchAgent (label `dev.kindling.cloudflared`):

```bash
cp ~/.kindling/cloudflared/launchd/dev.kindling.cloudflared.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.kindling.cloudflared.plist
launchctl kickstart -k gui/$(id -u)/dev.kindling.cloudflared     # restart after config changes
tail -f ~/.kindling/cloudflared/logs/cloudflared.err.log
```

Don't use `cloudflared service install`: it installs `com.cloudflare.cloudflared` reading `~/.cloudflared`
(or `/etc/cloudflared` with sudo), which would collide with the existing service on this Mac.
`dev.kindling.serve.plist` does the same for `kindling serve` if you want Kindling itself to start at login.

### 3. DNS

A proxied CNAME `kindling.example.com -> <tunnel-id>.cfargotunnel.com` (what `cloudflared tunnel route dns`
creates). The script won't overwrite a record that already exists and points elsewhere (`FORCE_DNS=1`
repoints a CNAME).

### 4. Cloudflare Access (optional, dashboard only)

Zero Trust > Access > Applications > Add an application > Self-hosted, hostname `kindling.example.com`.

- People: an **Allow** policy (emails, or your identity provider). Browsers get a login page.
- Machines: Zero Trust > Access > Service credentials > Service Tokens > Create, then a policy with action
  **Service Auth** that includes that token. Clients send `CF-Access-Client-Id` and `CF-Access-Client-Secret`
  on every request (or one header, if you set `read_service_tokens_from_header`).
- Access passes a signed JWT to the origin in `Cf-Access-Jwt-Assertion`; Kindling doesn't check it today.

Trade-off: with Access on the API hostname, **every** client must log in or send a service token, including
SDKs and Claude Code (both can add headers; see below). The gateway lane would have to forward the
`CF-Access-*` headers to the origin, which Cloudflare doesn't document **(unverified)**; if it doesn't, the
gateway lane stops working. Kindling keys already gate every API call, so the usual choice is:

- no Access on the API hostname, or
- Access on a **separate hostname** for Hearth's web UI (people log in), with the API hostname left to keys.

### 5. AI Gateway

Created by the script (`POST /accounts/{id}/ai-gateway/gateways`, updated with `PUT .../gateways/{id}`)
with authentication on (required for new gateways, and it can't be turned off), **logs off**, cache off,
and no gateway rate limit. Dashboard alternative: Compute & AI > AI Gateway > Create Gateway.

Then create the **client token**: in the gateway's dashboard page, create a token with Run permission (or an
API token with AI Gateway > Run). Save it; it isn't shown again.

### 6. Custom provider

Created by the script (`POST /accounts/{id}/ai-gateway/custom-providers`, updated with `PATCH
.../custom-providers/{id}`):

```json
{"name": "Kindling", "slug": "kindling", "base_url": "https://kindling.example.com", "enable": true}
```

Dashboard alternative: Compute & AI > AI Gateway > Custom Providers > Add Custom Provider. Custom providers
belong to the account, not a gateway, and `base_url` must be `https://`.

The gateway appends everything after `custom-kindling/` to `base_url`:

```
https://gateway.ai.cloudflare.com/v1/<acct>/<gw>/custom-kindling/v1/chat/completions
  -> https://kindling.example.com/v1/chat/completions
```

So keep `base_url` at the bare hostname and put `/v1` in the client's base URL. Cloudflare's **Unified API**
(`.../compat/chat/completions` with `model: "custom-kindling/<model>"`) also accepts custom providers, but
which upstream path it calls isn't documented **(unverified)**; use the provider-specific URL above.

### 7. Kindling settings

In Kindling's `.env`:

```bash
KINDLING_TRUST_CLOUDFLARE_HEADERS=true          # client IP from CF-Connecting-IP
KINDLING_ORIGIN_SECRET=<same value as ORIGIN_SECRET>   # only if you made the Transform Rule
# KINDLING_ORIGIN_SECRET_HEADER=X-Kindling-Origin-Secret  (the default)
KINDLING_CORS_ORIGINS=["https://hearth.example.com"]    # if Hearth is served from another origin
```

Keep `KINDLING_HOST=127.0.0.1`. Restart Kindling.

## Client configuration

Replace `<acct>`, `<gw>` and the keys. `kdl_...` is a Kindling key, `$CF_AIG_TOKEN` the gateway token.

### OpenAI SDK, through the gateway

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://gateway.ai.cloudflare.com/v1/<acct>/<gw>/custom-kindling/v1",
    api_key="kdl_...",  # forwarded to Kindling as Authorization: Bearer
    default_headers={
        "cf-aig-authorization": f"Bearer {CF_AIG_TOKEN}",
        "cf-aig-collect-log-payload": "false",  # keep metadata, not prompts, if logs are on
        "cf-aig-request-timeout": "120000",  # ms to the first byte; a cold model load can be slow
    },
)
reply = client.chat.completions.create(
    model="nemotron-3-nano:30b", messages=[{"role": "user", "content": "Hello"}], stream=True
)
```

Direct instead: `base_url="https://kindling.example.com/v1"`, no `default_headers`.

### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="https://kindling.example.com", api_key="kdl_...")  # sent as x-api-key
```

Through the gateway **(unverified)**: `base_url="https://gateway.ai.cloudflare.com/v1/<acct>/<gw>/custom-kindling"`,
`auth_token="kdl_..."` (so it goes in `Authorization`, the header the gateway treats as the provider key),
and `default_headers={"cf-aig-authorization": "Bearer <token>"}`.

### Claude Code

Direct to the tunnel hostname (recommended):

```bash
export ANTHROPIC_BASE_URL=https://kindling.example.com
export ANTHROPIC_AUTH_TOKEN=kdl_...          # sent as Authorization: Bearer
# If Kindling doesn't map Claude model names, pick one of its models (see /v1/models):
# export ANTHROPIC_MODEL=nemotron-3-nano:30b
```

Or put the same values in the `env` block of `~/.claude/settings.json`. With Access on the hostname add
`ANTHROPIC_CUSTOM_HEADERS=$'CF-Access-Client-Id: ...\nCF-Access-Client-Secret: ...'`.

Through the gateway (experimental, the Anthropic format through a custom provider is unverified):

```bash
export ANTHROPIC_BASE_URL=https://gateway.ai.cloudflare.com/v1/<acct>/<gw>/custom-kindling
export ANTHROPIC_AUTH_TOKEN=kdl_...
export ANTHROPIC_CUSTOM_HEADERS="cf-aig-authorization: Bearer $CF_AIG_TOKEN"
```

Cloudflare's own Claude Code guide uses the same `ANTHROPIC_CUSTOM_HEADERS` pattern for its built-in
`/anthropic` provider.

### Hearth

Point Hearth's API base URL at `https://kindling.example.com` and give it a Kindling key with only the
`chat` scope. If Hearth is served from a different origin, add that origin to `KINDLING_CORS_ORIGINS`.
Never point Hearth at the gateway (it would need a Cloudflare token in the browser).

### Automation tools

Anything with an "OpenAI-compatible" credential (n8n, LangChain, LiteLLM, curl in cron): base URL
`https://kindling.example.com/v1` with a Kindling key, or the gateway URL plus a custom
`cf-aig-authorization` header if the tool can add headers.

## Origin protection and client IPs

Kindling keys are always required; nothing here replaces them. The edge middleware
([src/kindling/edge.py](../src/kindling/edge.py)) adds two opt-in checks.

### Origin secret (`KINDLING_ORIGIN_SECRET`)

When set, every request must carry exactly one `X-Kindling-Origin-Secret` header with that value (compared in
constant time), or it gets `403 {"detail": "forbidden"}`. The header is removed before the app sees it.

- **Exempt:** `/livez`, `/readyz` and `/api/health`, so probes on the Mac keep working. They need no key
  and reveal no secrets anyway.
- **Who adds it:** a Cloudflare **Transform Rule** (Rules > Transform Rules > Modify Request Header >
  Set static, `X-Kindling-Origin-Secret`, when hostname equals `kindling.example.com`). `setup.sh` creates it
  when `ORIGIN_SECRET` is set. Transform Rules are on every plan (Free: 10 rules).
- **Why not the other options:**
  - AI Gateway custom providers have no setting for extra upstream headers (the API fields are name, slug,
    base_url, description, link, enable and a few display fields), and it would only cover the gateway lane.
  - Access service tokens are sent by the client, not added by Cloudflare; Access gives the origin a JWT
    instead, which Kindling would need code to verify.
  - The Transform Rule covers both lanes, since the gateway calls the public hostname. That gateway requests
    pass through the zone's Transform Rules is expected but **unverified**: test the gateway lane (curl
    below) right after turning the secret on.
- **What it protects against:** with a tunnel, the port is only reachable from the Mac, so this is about
  traffic that didn't come through *your* hostname: another program or user on the Mac, a browser page
  making requests to `127.0.0.1:8000`, or another cloudflared on the same Mac (the organisation's) routing to
  the port. It also makes `TRUST_CLOUDFLARE_HEADERS` spoof-proof.
- **Cost:** local clients that call `http://127.0.0.1:8000` directly (Hearth in dev, Claude Code on the Mac)
  now need the header too, or should use the public hostname. Claude Code:
  `ANTHROPIC_CUSTOM_HEADERS="X-Kindling-Origin-Secret: ..."`.
- Rotate it by changing the rule and `.env` together (requests fail with 403 between the two).

### Client IP (`KINDLING_TRUST_CLOUDFLARE_HEADERS`)

Through a tunnel every request comes from cloudflared on `127.0.0.1`. With this on, Kindling takes the client
address from `CF-Connecting-IP` (if it's a valid IP), so logs and anything keyed on IP see the real caller.

- Only safe when the server is reachable **only through Cloudflare**: bound to `127.0.0.1` behind the tunnel.
  Anyone who can reach the port directly can put any address in that header. Kindling logs a warning if this
  is on, there is no origin secret, and it isn't bound to loopback.
- With an origin secret too, the header is trusted only on requests whose secret matched.
- Through the gateway lane, `CF-Connecting-IP` is the gateway's address, not the end client's **(unverified)**.
- Kindling runs uvicorn with `proxy_headers=False`, so `X-Forwarded-For` is never trusted.

## Gateway features: use, avoid, costs

| Feature | For Kindling chat | Why |
|---|---|---|
| Authentication | required | can't be turned off on new gateways |
| Logs | **off** by default in `setup.sh` | Gateway logs store full prompts and replies at Cloudflare. If you want analytics, turn logs on and send `cf-aig-collect-log-payload: false` (metadata only), or `cf-aig-collect-log: false` per request. |
| Caching | **avoid** | Exact match on the whole body and the auth header, so chats with history rarely hit; a hit replays an old answer, which is wrong for tool calls and sampling. Off by default. If you cache something deliberately, opt in per request with `cf-aig-cache-ttl` / `cf-aig-cache-key`. |
| Rate limiting | avoid | One limit for the whole gateway, shared by every caller. Kindling's per-key limits are better. |
| Retries | avoid for generation | A retried generation runs the model again on the Mac. Max 5 attempts, delay up to 60 s. |
| Request timeout | use | `cf-aig-request-timeout` (ms) is measured to the first part of the response; once streaming starts the gateway waits for the rest. |
| Fallbacks | maybe | Universal endpoint / dynamic routes can fall back to another provider when Kindling is down; whether a custom provider can be the primary there is **unverified**. Remember a fallback sends your prompt to someone else. |
| DLP, guardrails | optional | Guardrails are billed as Workers AI inference. |

**Costs.** AI Gateway's core features (analytics, caching, rate limiting) are free. Accounts whose first
gateway was created on or after 24 September 2026 store logs in Workers Logs, with Workers Logs pricing and
retention; older ones use Legacy Logs (Free: 100,000 logs per account, Paid: 10 million per gateway). Logpush
is billed separately. Gateways: 10 per account on Free, 20 on Paid. Tunnel itself has no charge. Access is
part of Cloudflare Zero Trust (its free tier covers small teams; check current Zero Trust pricing).

## Timeouts and long responses

- **Cloudflare's Proxy Read Timeout is 125 seconds** (older pages say 100). If Kindling sends nothing for
  that long the visitor gets **HTTP 524**. Only Enterprise can raise it. Cloudflare's 524 page doesn't say
  whether the clock restarts on every chunk, but the practical rule is: the first byte must arrive within
  125 s, and a stream must keep producing bytes.
- **Non-streaming replies are padded.** If a non-streaming `/v1/chat/completions` or `/v1/messages` reply
  isn't ready after 20 s, Kindling sends status 200 and the headers, then a space every 15 s, then the JSON
  (leading whitespace is valid JSON, so SDKs parse it as usual). A slow reply therefore isn't cut off with a
  524. The cost: an error after that point arrives as the API's error body with status 200, so a client
  that only checks the status code should also check for an `error` field. Streaming (`"stream": true`)
  remains the better choice for long replies: the text arrives as it is written.
- **Reasoning phases.** Nemotron can think for a long time before its answer. If the thinking tokens are
  streamed, bytes keep flowing. If an endpoint holds them back (or the model is loading, or a request waits
  in Kindling's generation queue for up to `KINDLING_QUEUE_TIMEOUT_SECONDS`), the connection can sit silent
  past 125 s. Sending response headers at once and an SSE comment line (`: keepalive`) every 15 to 30 s
  while waiting keeps it alive.
- **Cold starts.** Ollama loading a 30B model can take tens of seconds; warm it (`kindling check`) after a
  restart. Through the gateway, set `cf-aig-request-timeout` high enough (e.g. `120000`); its default isn't
  documented.
- **Request upload:** Cloudflare's proxy write timeout is 30 s and can't be changed; irrelevant for chat-sized
  bodies (Kindling caps them at 2 MB).
- **Buffering.** If chunks arrive in bursts instead of smoothly, check with `curl -N` against the hostname;
  SSE (`text/event-stream`) is passed through as it arrives. Whether Cloudflare's compression buffers
  NDJSON responses (`/api/chat`) is **unverified**; a Compression Rule that disables compression for
  `/api/chat` fixes it if so.

## Troubleshooting: check each hop

Run these in order; the first one that fails is the broken hop.

```bash
H=kindling.example.com; K=kdl_...; ACCT=<acct>; GW=kindling; T=$CF_AIG_TOKEN

# 1. Ollama
curl -s http://127.0.0.1:11434/api/tags | head -c 300

# 2. Kindling, locally (add -H "X-Kindling-Origin-Secret: ..." if the origin secret is on)
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $K"

# 3. cloudflared: is it connected?
curl -s http://127.0.0.1:20241/ready          # metrics address from config.yml
cloudflared tunnel --config ~/.kindling/cloudflared/config.yml ingress rule https://$H

# 4. DNS and the edge
dig +short $H                                  # Cloudflare IPs (proxied), not cfargotunnel.com
curl -sI https://$H/api/health                 # 200, server: cloudflare

# 5. Through the tunnel, with a key (checks auth, and the Transform Rule if the secret is on)
curl -s https://$H/v1/models -H "Authorization: Bearer $K"
curl -sN https://$H/v1/chat/completions -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"model":"nemotron-3-nano:30b","stream":true,"messages":[{"role":"user","content":"Say hi"}]}'

# 6. Through the gateway
curl -s https://gateway.ai.cloudflare.com/v1/$ACCT/$GW/custom-kindling/v1/models \
  -H "cf-aig-authorization: Bearer $T" -H "Authorization: Bearer $K"
curl -sN https://gateway.ai.cloudflare.com/v1/$ACCT/$GW/custom-kindling/v1/chat/completions \
  -H "cf-aig-authorization: Bearer $T" -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"model":"nemotron-3-nano:30b","stream":true,"messages":[{"role":"user","content":"Say hi"}]}'

# 7. Anthropic format through the gateway (the unverified part)
curl -sN https://gateway.ai.cloudflare.com/v1/$ACCT/$GW/custom-kindling/v1/messages \
  -H "cf-aig-authorization: Bearer $T" -H "Authorization: Bearer $K" \
  -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" \
  -d '{"model":"nemotron-3-nano:30b","max_tokens":200,"stream":true,"messages":[{"role":"user","content":"Say hi"}]}'
```

| Symptom | Likely cause |
|---|---|
| 1033 / 530 from Cloudflare | cloudflared isn't running or isn't connected; check its log in `~/.kindling/cloudflared/logs/` |
| 502 from Cloudflare | cloudflared is up but Kindling isn't listening on `127.0.0.1:8000` |
| 404 with an empty body | the request hit the catch-all: the hostname in `config.yml` doesn't match |
| 403 `{"detail":"forbidden"}` | origin secret missing or wrong (Transform Rule vs `.env`), or you called `127.0.0.1` without it |
| 401 from Kindling | missing or bad Kindling key (through the gateway: the key must be in `Authorization: Bearer`) |
| 401 from the gateway | missing or bad `cf-aig-authorization` token |
| 524 | nothing sent for 125 s: on /v1 Kindling pads and pings, so look at the hop in between (a gateway, a proxy that buffers), warm the model, or shorten the queue |
| Access login page in an SDK | Access is on the hostname: add the service token headers |
| `tunnel credentials file ... doesn't exist` | the tunnel was created elsewhere; restore the JSON or delete the tunnel and re-run `setup.sh` |

## What was verified, with sources

Verified in Cloudflare's docs (November 2026):

- Custom providers: `POST/GET /accounts/{id}/ai-gateway/custom-providers`, `PATCH .../custom-providers/{id}`;
  fields `name`, `slug`, `base_url` (HTTPS only), `description`, `link`, `enable`, `beta`; URL
  `.../{gateway_id}/custom-{slug}/{path}` maps to `{base_url}/{path}`; provider key in
  `Authorization: Bearer`; gateway token in `cf-aig-authorization`; Unified API supports
  `custom-{slug}/{model}` for chat completions only.
  [Custom providers](https://developers.cloudflare.com/ai-gateway/configuration/custom-providers/)
- New gateways must be authenticated, one-way; Run tokens can't be scoped to one gateway.
  [Authentication](https://developers.cloudflare.com/ai-gateway/configuration/authentication/)
- BYOK keys live in Secrets Store and need an authenticated gateway; a client `Authorization` header wins.
  [BYOK](https://developers.cloudflare.com/ai-gateway/configuration/bring-your-own-keys/)
- Gateway create/update fields (`cache_ttl`, `collect_logs`, `rate_limiting_*`, `authentication`,
  `retry_*`, `zdr`, ...). [Create](https://developers.cloudflare.com/api/resources/ai_gateway/methods/create/),
  [Update](https://developers.cloudflare.com/api/resources/ai_gateway/methods/update/)
- Caching off by default, exact match, TTL 60 s to 1 month.
  [Caching](https://developers.cloudflare.com/ai-gateway/features/caching/)
- Logs on by default and include prompts and responses; `cf-aig-collect-log`, `cf-aig-collect-log-payload`.
  [Logging](https://developers.cloudflare.com/ai-gateway/observability/logging/)
- Rate limits are per gateway. [Rate limiting](https://developers.cloudflare.com/ai-gateway/features/rate-limiting/)
- Timeout to first byte, retries up to 5. [Request handling](https://developers.cloudflare.com/ai-gateway/configuration/request-handling/)
- Fallbacks through the Universal endpoint. [Fallbacks](https://developers.cloudflare.com/ai-gateway/configuration/fallbacks/)
- Limits and pricing. [Limits](https://developers.cloudflare.com/ai-gateway/reference/limits/),
  [Pricing](https://developers.cloudflare.com/ai-gateway/reference/pricing/)
- Claude Code with `ANTHROPIC_CUSTOM_HEADERS` for `cf-aig-authorization` (built-in Anthropic provider).
  [Claude Code](https://developers.cloudflare.com/ai-gateway/integrations/coding-agents/claude-code/)
- Tunnels: [config file and ingress](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/configuration-file/),
  [locally managed tunnel, `route dns`](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/),
  [API: create tunnel](https://developers.cloudflare.com/api/resources/zero_trust/subresources/tunnels/subresources/cloudflared/methods/create/),
  [API tunnel walkthrough (DNS record, permissions)](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel-api/),
  [macOS service](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/as-a-service/macos/)
- Access: [service tokens](https://developers.cloudflare.com/cloudflare-one/access-controls/service-credentials/service-tokens/),
  [validating the Access JWT](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/)
- Transform Rules: [request header modification](https://developers.cloudflare.com/rules/transform/request-header-modification/),
  [via API](https://developers.cloudflare.com/rules/transform/request-header-modification/create-api/),
  [plans](https://developers.cloudflare.com/rules/transform/)
- 524 and the 125 s Proxy Read Timeout. [Error 524](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/)

Not confirmed (test with the curl commands above):

- The Anthropic Messages format (`/v1/messages`) through a custom provider, and which request headers
  (`anthropic-version`, `x-api-key`, `CF-Access-*`) the gateway forwards to a custom provider.
- BYOK stored keys for custom providers.
- The upstream path the Unified API (`/compat`) uses for custom providers.
- Whether gateway requests to the tunnel hostname pass through the zone's Transform Rules (needed for the
  origin secret on the gateway lane).
- The default `cf-aig-request-timeout`, and whether the 125 s timeout resets on each streamed chunk.
- The gateway's `CF-Connecting-IP` as seen by Kindling.
- Exact single-gateway `GET` and 404 behaviour used by `setup.sh` to detect an existing gateway, and that
  `PUT .../rulesets/phases/http_request_late_transform/entrypoint` creates the phase ruleset when missing.
- The credentials file format (`AccountTag`, `TunnelSecret`, `TunnelID`) for an API-created locally managed
  tunnel is the one `cloudflared tunnel create` writes; it isn't on the pages above.
