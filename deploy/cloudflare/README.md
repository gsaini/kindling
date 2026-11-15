# deploy/cloudflare

Puts Kindling on the internet through Cloudflare Tunnel, with an optional AI Gateway in front.
The full guide (topology, keys, client setup, timeouts, troubleshooting) is
[docs/cloudflare.md](../../docs/cloudflare.md).

| File | What it is |
|---|---|
| `setup.sh` | Creates or updates the tunnel, DNS record, local config, AI Gateway, custom provider and (optionally) the origin secret Transform Rule. **Dry run unless you pass `--apply`.** |
| `config.yml.template` | cloudflared config: `kindling.example.com` to `http://127.0.0.1:8000`, 404 for anything else. |
| `launchd/dev.kindling.cloudflared.plist.template` | Runs the tunnel at login (LaunchAgent, as you). |
| `launchd/dev.kindling.serve.plist.template` | Optional: runs `kindling serve` at login. |

## Quick start

```bash
# 1. See what it would do (no network calls, no files written):
CF_ACCOUNT_ID=... KINDLING_HOSTNAME=kindling.example.com ./deploy/cloudflare/setup.sh

# 2. Do it:
export CF_ACCOUNT_ID=... CF_API_TOKEN=... KINDLING_HOSTNAME=kindling.example.com GATEWAY_ID=kindling
./deploy/cloudflare/setup.sh --apply

# 3. Run the tunnel:
cloudflared tunnel --config ~/.kindling/cloudflared/config.yml run
```

`--no-tunnel` and `--no-gateway` skip either half. `./setup.sh --help` lists every variable.

## API token permissions

Create a custom API token on the account Kindling will use (My Profile > API Tokens > Create Token):

| Permission | Why |
|---|---|
| Account > Cloudflare Tunnel > Edit | create the tunnel |
| Account > AI Gateway > Edit | create the gateway and the custom provider |
| Zone > DNS > Edit (your zone) | the CNAME to the tunnel |
| Zone > Zone > Read (your zone) | find the zone id (or set `CF_ZONE_ID`) |
| Zone > Transform Rules > Edit (your zone) | only with `ORIGIN_SECRET` |

Clients that go through the gateway need a different token, with only **AI Gateway > Run**. Note that a
Run token works for *every* gateway on the account, which is one reason to keep Kindling on its own account.

## What it touches, and what it doesn't

- Everything local lives in `KINDLING_CF_DIR` (default `~/.kindling/cloudflared`): the tunnel credentials
  (`<tunnel-id>.json`, mode 600), `config.yml`, rendered plists and logs.
- It never runs `cloudflared login` or `cloudflared service install`, and never reads or writes
  `~/.cloudflared` or `/etc/cloudflared`, which may belong to a different Cloudflare account. The tunnel is
  run by UUID with a credentials file, so no `cert.pem` is needed. The launchd label is
  `dev.kindling.cloudflared`, not `com.cloudflare.cloudflared`, so it sits beside any existing cloudflared service.
- Re-running is safe. An existing tunnel is reused (its credentials file must still be in `KINDLING_CF_DIR`;
  the secret can't be downloaded again). A DNS record that points somewhere else is left alone unless
  `FORCE_DNS=1`. An existing gateway gets the script's settings (logs off, cache off, no gateway rate limit
  unless you set `GATEWAY_COLLECT_LOGS` / `GATEWAY_CACHE_TTL`), so dashboard changes to those are reset.
  The Transform Rule is found by its description and updated; other rules in the zone are untouched.
- Cloudflare Access isn't scripted: set it up in the dashboard if you want it (see the guide).

## Removing it

Dashboard: delete the DNS record, the tunnel (Zero Trust > Networks > Tunnels), the custom provider and the
gateway (AI > AI Gateway), and the Transform Rule. Locally: `launchctl bootout gui/$(id -u)/dev.kindling.cloudflared`,
remove the plist from `~/Library/LaunchAgents`, and delete `~/.kindling/cloudflared`.
