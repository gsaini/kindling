#!/usr/bin/env bash
# Set up Cloudflare in front of Kindling: a named tunnel, its DNS record, the local cloudflared
# config, launchd plists, an AI Gateway with a custom provider pointing at the tunnel hostname, and
# (optionally) a Transform Rule that adds the origin secret header.
#
# Dry run by default: prints every request it would make and changes nothing, makes no network
# calls and writes no files. Pass --apply to do it. Safe to re-run: each step looks before it acts.
#
# Uses only the Cloudflare API with CF_API_TOKEN. It never runs `cloudflared login`, never reads
# ~/.cloudflared, and keeps its own files in KINDLING_CF_DIR.
#
# Required (for --apply):
#   CF_ACCOUNT_ID        the Cloudflare account that owns the zone
#   CF_API_TOKEN         an API token (permissions: see README.md)
#   KINDLING_HOSTNAME    e.g. kindling.example.com, in a zone on that account
# Optional:
#   GATEWAY_ID           AI Gateway id (default: kindling)
#   CF_ZONE_ID           skip the zone lookup (then the token needs no Zone:Read)
#   TUNNEL_NAME          default: kindling
#   PROVIDER_SLUG        custom provider slug (default: kindling; clients call custom-<slug>/...)
#   KINDLING_ORIGIN      where cloudflared sends traffic (default: http://127.0.0.1:8000)
#   KINDLING_CF_DIR      local files (default: ~/.kindling/cloudflared)
#   KINDLING_REPO        repo path for the kindling launchd plist (default: this checkout)
#   ORIGIN_SECRET        if set, add a Transform Rule that sends it in ORIGIN_SECRET_HEADER
#   ORIGIN_SECRET_HEADER default: X-Kindling-Origin-Secret (must match KINDLING_ORIGIN_SECRET_HEADER)
#   GATEWAY_COLLECT_LOGS true/false (default: false; logs store prompts and replies at Cloudflare)
#   GATEWAY_CACHE_TTL    seconds, 0 = off (default: 0)
#   FORCE_DNS=1          repoint an existing CNAME for KINDLING_HOSTNAME at this tunnel
#
# Flags: --apply, --no-tunnel (skip tunnel, DNS, config), --no-gateway, --help

set -euo pipefail

APPLY=0
DO_TUNNEL=1
DO_GATEWAY=1
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --no-tunnel) DO_TUNNEL=0 ;;
    --no-gateway) DO_GATEWAY=0 ;;
    -h | --help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="https://api.cloudflare.com/client/v4"

CF_ACCOUNT_ID="${CF_ACCOUNT_ID:-}"
CF_API_TOKEN="${CF_API_TOKEN:-}"
KINDLING_HOSTNAME="${KINDLING_HOSTNAME:-}"
GATEWAY_ID="${GATEWAY_ID:-kindling}"
CF_ZONE_ID="${CF_ZONE_ID:-}"
TUNNEL_NAME="${TUNNEL_NAME:-kindling}"
PROVIDER_SLUG="${PROVIDER_SLUG:-kindling}"
KINDLING_ORIGIN="${KINDLING_ORIGIN:-http://127.0.0.1:8000}"
KINDLING_CF_DIR="${KINDLING_CF_DIR:-$HOME/.kindling/cloudflared}"
KINDLING_REPO="${KINDLING_REPO:-$(cd "$HERE/../.." && pwd)}"
ORIGIN_SECRET="${ORIGIN_SECRET:-}"
ORIGIN_SECRET_HEADER="${ORIGIN_SECRET_HEADER:-X-Kindling-Origin-Secret}"
GATEWAY_COLLECT_LOGS="${GATEWAY_COLLECT_LOGS:-false}"
GATEWAY_CACHE_TTL="${GATEWAY_CACHE_TTL:-0}"
FORCE_DNS="${FORCE_DNS:-0}"

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }
die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

# ---- checks --------------------------------------------------------------------------------------

if [[ $APPLY -eq 1 ]]; then
  for name in CF_ACCOUNT_ID CF_API_TOKEN KINDLING_HOSTNAME; do
    [[ -n "${!name}" ]] || die "$name is required with --apply"
  done
  for tool in curl jq openssl; do
    command -v "$tool" >/dev/null || die "$tool is needed"
  done
else
  # A dry run shows the plan even before you have the values.
  [[ -n "$CF_ACCOUNT_ID" ]] || CF_ACCOUNT_ID="<CF_ACCOUNT_ID>"
  [[ -n "$KINDLING_HOSTNAME" ]] || KINDLING_HOSTNAME="<KINDLING_HOSTNAME>"
fi

if [[ $APPLY -eq 1 || "$KINDLING_HOSTNAME" != "<KINDLING_HOSTNAME>" ]]; then
  [[ "$KINDLING_HOSTNAME" =~ ^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$ ]] ||
    die "KINDLING_HOSTNAME should be a lowercase hostname like kindling.example.com (got '$KINDLING_HOSTNAME')"
fi
[[ "$GATEWAY_ID" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || die "GATEWAY_ID: lowercase letters, digits, - and _ only"
[[ "$PROVIDER_SLUG" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || die "PROVIDER_SLUG: letters, digits and hyphens only"
[[ "$TUNNEL_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "TUNNEL_NAME: letters, digits, . _ - only"
[[ "$GATEWAY_COLLECT_LOGS" =~ ^(true|false)$ ]] || die "GATEWAY_COLLECT_LOGS must be true or false"
[[ "$GATEWAY_CACHE_TTL" =~ ^[0-9]+$ ]] || die "GATEWAY_CACHE_TTL must be a number of seconds"
[[ "$ORIGIN_SECRET_HEADER" =~ ^[A-Za-z0-9-]+$ ]] || die "ORIGIN_SECRET_HEADER: letters, digits and hyphens only"
if [[ -n "$ORIGIN_SECRET" && ${#ORIGIN_SECRET} -lt 32 ]]; then
  die "ORIGIN_SECRET is short; make one with: openssl rand -hex 32"
fi

# Plain paths only, so they go into YAML and plist XML without quoting or escaping.
for name in KINDLING_CF_DIR KINDLING_REPO; do
  [[ "${!name}" =~ ^/[A-Za-z0-9._/\ -]+$ ]] || die "$name should be an absolute path of letters, digits, spaces and ._-/"
done
[[ "$KINDLING_ORIGIN" =~ ^https?://[A-Za-z0-9.:-]+/?$ ]] || die "KINDLING_ORIGIN should look like http://127.0.0.1:8000"

# ~/.cloudflared may hold another account's credentials: never write there.
case "$KINDLING_CF_DIR/" in
  "$HOME/.cloudflared/"* | "/etc/cloudflared/"*) die "KINDLING_CF_DIR must not be inside ~/.cloudflared or /etc/cloudflared" ;;
esac

if [[ $APPLY -eq 1 ]]; then
  echo "Applying changes to Cloudflare account $CF_ACCOUNT_ID for $KINDLING_HOSTNAME."
else
  echo "DRY RUN: nothing will be changed and no requests sent. Re-run with --apply to do it."
fi

# ---- the Cloudflare API --------------------------------------------------------------------------

AUTH_FILE=""
cleanup() { if [[ -n "$AUTH_FILE" ]]; then rm -f "$AUTH_FILE"; fi; }
trap cleanup EXIT

# The token goes to curl through a private file, so it never shows up in `ps`.
auth_header_file() {
  if [[ -z "$AUTH_FILE" ]]; then
    AUTH_FILE="$(mktemp)"
    chmod 600 "$AUTH_FILE"
    printf 'Authorization: Bearer %s\n' "$CF_API_TOKEN" >"$AUTH_FILE"
  fi
}

redact() {
  local text="$1"
  [[ -n "$ORIGIN_SECRET" ]] && text="${text//"$ORIGIN_SECRET"/<ORIGIN_SECRET>}"
  printf '%s' "$text" | sed -E 's/("tunnel_secret": *")[^"]*"/\1<generated>"/'
}

# plan METHOD PATH [BODY]: print a request.
plan() {
  note "$1 $API$2"
  if [[ -n "${3:-}" ]]; then
    redact "$3" | sed 's/^/        /'
    echo
  fi
}

# cf METHOD PATH [BODY]: send a request; sets CF_STATUS and CF_BODY. Only used with --apply.
cf() {
  local method="$1" path="$2" body="${3:-}" out
  auth_header_file
  plan "$method" "$path" "$body"
  local args=(-sS -X "$method" -H "@$AUTH_FILE" -H "Content-Type: application/json" -w $'\n%{http_code}')
  [[ -n "$body" ]] && args+=(--data "$body")
  out="$(curl "${args[@]}" "$API$path")" || die "curl failed: $method $path"
  CF_STATUS="${out##*$'\n'}"
  CF_BODY="${out%$'\n'*}"
}

# cf_ok: the last call succeeded, or stop with Cloudflare's error messages.
cf_ok() {
  if [[ "$CF_STATUS" != 2* ]] || [[ "$(jq -r '.success' <<<"$CF_BODY" 2>/dev/null)" != "true" ]]; then
    local errors
    errors="$(jq -r '.errors[]? | "\(.code): \(.message)"' <<<"$CF_BODY" 2>/dev/null || true)"
    die "Cloudflare returned HTTP $CF_STATUS: ${errors:-$CF_BODY}"
  fi
}

# replace_all TEXT TOKEN VALUE: TEXT with every TOKEN replaced. Done by hand because ${x//a/b}
# treats quotes and & differently in macOS's bash 3.2 and in bash 5.2.
replace_all() {
  local text="$1" token="$2" value="$3" out=""
  while [[ "$text" == *"$token"* ]]; do
    out="$out${text%%"$token"*}$value"
    text="${text#*"$token"}"
  done
  printf '%s' "$out$text"
}

# render TEMPLATE: the template with ${NAME} placeholders filled, on stdout.
render() {
  local line name
  while IFS= read -r line || [[ -n "$line" ]]; do
    for name in TUNNEL_ID KINDLING_CF_DIR KINDLING_HOSTNAME KINDLING_ORIGIN KINDLING_REPO CLOUDFLARED_BIN UV_BIN; do
      line="$(replace_all "$line" "\${$name}" "${!name}")"
    done
    printf '%s\n' "$line"
  done <"$1"
}

# write_if_changed DEST MODE: stdin to DEST, keeping a .bak of a different old version.
write_if_changed() {
  local dest="$1" mode="$2" tmp
  tmp="$(mktemp)"
  cat >"$tmp"
  if [[ -f "$dest" ]] && cmp -s "$tmp" "$dest"; then
    rm -f "$tmp"
    note "unchanged: $dest"
    return
  fi
  [[ -f "$dest" ]] && cp -p "$dest" "$dest.bak" && note "previous version kept as $dest.bak"
  mkdir -p "$(dirname "$dest")"
  mv "$tmp" "$dest"
  chmod "$mode" "$dest"
  note "wrote $dest"
}

TUNNEL_ID="<tunnel-id>"
CLOUDFLARED_BIN="$(command -v cloudflared || echo /opt/homebrew/bin/cloudflared)"
UV_BIN="$(command -v uv || echo /opt/homebrew/bin/uv)"

# ---- 1. tunnel -----------------------------------------------------------------------------------

step_tunnel() {
  say "1. Named tunnel '$TUNNEL_NAME' (locally managed: ingress lives in config.yml)"
  local list="/accounts/$CF_ACCOUNT_ID/cfd_tunnel?name=$TUNNEL_NAME&is_deleted=false"
  local create="/accounts/$CF_ACCOUNT_ID/cfd_tunnel"
  if [[ $APPLY -eq 0 ]]; then
    plan GET "$list"
    note "if there is none:"
    plan POST "$create" "{\"name\": \"$TUNNEL_NAME\", \"config_src\": \"local\", \"tunnel_secret\": \"<generated>\"}"
    note "and save its credentials to $KINDLING_CF_DIR/<tunnel-id>.json (mode 600)"
    return
  fi

  cf GET "$list"
  cf_ok
  TUNNEL_ID="$(jq -r --arg n "$TUNNEL_NAME" '[.result[] | select(.name == $n)][0].id // empty' <<<"$CF_BODY")"
  if [[ -n "$TUNNEL_ID" ]]; then
    note "exists: $TUNNEL_ID"
    [[ -f "$KINDLING_CF_DIR/$TUNNEL_ID.json" ]] ||
      die "tunnel '$TUNNEL_NAME' exists but $KINDLING_CF_DIR/$TUNNEL_ID.json doesn't, and its secret can't be fetched back. Either restore that file, delete the tunnel in the dashboard, or pick another TUNNEL_NAME."
    return
  fi

  local secret body
  secret="$(openssl rand -base64 32)"
  body="$(jq -n --arg n "$TUNNEL_NAME" --arg s "$secret" '{name: $n, config_src: "local", tunnel_secret: $s}')"
  cf POST "$create" "$body"
  cf_ok
  TUNNEL_ID="$(jq -r '.result.id' <<<"$CF_BODY")"
  note "created: $TUNNEL_ID"
  mkdir -p "$KINDLING_CF_DIR"
  chmod 700 "$KINDLING_CF_DIR"
  jq -n --arg a "$CF_ACCOUNT_ID" --arg s "$secret" --arg t "$TUNNEL_ID" \
    '{AccountTag: $a, TunnelSecret: $s, TunnelID: $t}' | write_if_changed "$KINDLING_CF_DIR/$TUNNEL_ID.json" 600
}

# ---- 2. config.yml and launchd plists ------------------------------------------------------------

step_config() {
  say "2. cloudflared config and launchd plists in $KINDLING_CF_DIR"
  if [[ $APPLY -eq 0 ]]; then
    note "would write $KINDLING_CF_DIR/config.yml:"
    render "$HERE/config.yml.template" | grep -v '^#' | sed 's/^/        /'
    note "would write $KINDLING_CF_DIR/launchd/dev.kindling.cloudflared.plist and dev.kindling.serve.plist"
    return
  fi
  mkdir -p "$KINDLING_CF_DIR/logs" "$KINDLING_CF_DIR/launchd"
  render "$HERE/config.yml.template" | write_if_changed "$KINDLING_CF_DIR/config.yml" 600
  for name in dev.kindling.cloudflared dev.kindling.serve; do
    render "$HERE/launchd/$name.plist.template" | write_if_changed "$KINDLING_CF_DIR/launchd/$name.plist" 644
  done
  if "$CLOUDFLARED_BIN" tunnel --config "$KINDLING_CF_DIR/config.yml" ingress validate >/dev/null 2>&1; then
    note "cloudflared says the ingress rules are valid"
  else
    warn "cloudflared tunnel ingress validate failed; check $KINDLING_CF_DIR/config.yml"
  fi
}

# ---- 3. zone and DNS -----------------------------------------------------------------------------

find_zone() {
  [[ -n "$CF_ZONE_ID" ]] && return
  # Try kindling.example.com, then example.com, ... until a zone on the account matches.
  local candidate="$KINDLING_HOSTNAME"
  while [[ "$candidate" == *.* ]]; do
    cf GET "/zones?name=$candidate&account.id=$CF_ACCOUNT_ID"
    cf_ok
    CF_ZONE_ID="$(jq -r '.result[0].id // empty' <<<"$CF_BODY")"
    if [[ -n "$CF_ZONE_ID" ]]; then
      note "zone: $candidate ($CF_ZONE_ID)"
      return
    fi
    candidate="${candidate#*.}"
  done
  die "no zone on account $CF_ACCOUNT_ID contains $KINDLING_HOSTNAME (or set CF_ZONE_ID)"
}

step_dns() {
  say "3. DNS: $KINDLING_HOSTNAME -> CNAME <tunnel-id>.cfargotunnel.com (proxied)"
  local target="$TUNNEL_ID.cfargotunnel.com"
  if [[ $APPLY -eq 0 ]]; then
    [[ -n "$CF_ZONE_ID" ]] || plan GET "/zones?name=<parent domain of $KINDLING_HOSTNAME>&account.id=$CF_ACCOUNT_ID"
    plan GET "/zones/${CF_ZONE_ID:-<zone-id>}/dns_records?name=$KINDLING_HOSTNAME"
    note "if there is none:"
    plan POST "/zones/${CF_ZONE_ID:-<zone-id>}/dns_records" \
      "{\"type\": \"CNAME\", \"name\": \"$KINDLING_HOSTNAME\", \"content\": \"$target\", \"proxied\": true}"
    note "an existing record pointing elsewhere is left alone (FORCE_DNS=1 repoints a CNAME)"
    return
  fi

  find_zone
  cf GET "/zones/$CF_ZONE_ID/dns_records?name=$KINDLING_HOSTNAME"
  cf_ok
  local count type content proxied id
  count="$(jq '.result | length' <<<"$CF_BODY")"
  local body
  body="$(jq -n --arg n "$KINDLING_HOSTNAME" --arg c "$target" \
    '{type: "CNAME", name: $n, content: $c, proxied: true, comment: "Kindling tunnel (deploy/cloudflare/setup.sh)"}')"
  if [[ "$count" -eq 0 ]]; then
    cf POST "/zones/$CF_ZONE_ID/dns_records" "$body"
    cf_ok
    note "created"
    return
  fi
  type="$(jq -r '.result[0].type' <<<"$CF_BODY")"
  content="$(jq -r '.result[0].content' <<<"$CF_BODY")"
  proxied="$(jq -r '.result[0].proxied' <<<"$CF_BODY")"
  id="$(jq -r '.result[0].id' <<<"$CF_BODY")"
  if [[ "$count" -eq 1 && "$type" == CNAME && "$content" == "$target" && "$proxied" == true ]]; then
    note "already correct"
  elif [[ "$count" -eq 1 && "$type" == CNAME && "$FORCE_DNS" == 1 ]]; then
    cf PATCH "/zones/$CF_ZONE_ID/dns_records/$id" "$body"
    cf_ok
    note "repointed (was $content)"
  else
    die "$KINDLING_HOSTNAME already has $count record(s) ($type -> $content). Not touching it; pick another hostname, or FORCE_DNS=1 to repoint a CNAME."
  fi
}

# ---- 4. origin secret Transform Rule (optional) --------------------------------------------------

RULE_DESCRIPTION="Kindling origin secret (deploy/cloudflare/setup.sh)"

step_origin_rule() {
  [[ -n "$ORIGIN_SECRET" ]] || return 0
  say "4. Transform Rule: add $ORIGIN_SECRET_HEADER to requests for $KINDLING_HOSTNAME"
  local phase="/zones/${CF_ZONE_ID:-<zone-id>}/rulesets/phases/http_request_late_transform/entrypoint"
  local rule
  rule="$(jq -n --arg d "$RULE_DESCRIPTION" --arg h "$KINDLING_HOSTNAME" --arg name "$ORIGIN_SECRET_HEADER" \
    --arg v "$ORIGIN_SECRET" \
    '{description: $d, expression: "(http.host eq \"\($h)\")", action: "rewrite", enabled: true,
      action_parameters: {headers: {($name): {operation: "set", value: $v}}}}')"
  if [[ $APPLY -eq 0 ]]; then
    plan GET "$phase"
    note "if the phase has no ruleset yet:"
    plan PUT "$phase" "{\"rules\": [$rule]}"
    note "else add the rule (or update ours, found by its description):"
    plan POST "/zones/<zone-id>/rulesets/<ruleset-id>/rules" "$rule"
    note "then set KINDLING_ORIGIN_SECRET (same value) in Kindling's .env"
    return
  fi

  find_zone
  phase="/zones/$CF_ZONE_ID/rulesets/phases/http_request_late_transform/entrypoint"
  cf GET "$phase"
  if [[ "$CF_STATUS" == 404 ]]; then
    cf PUT "$phase" "$(jq -n --argjson r "$rule" '{rules: [$r]}')"
    cf_ok
    note "created the phase ruleset with our rule"
    return
  fi
  cf_ok
  local ruleset rule_id
  ruleset="$(jq -r '.result.id' <<<"$CF_BODY")"
  rule_id="$(jq -r --arg d "$RULE_DESCRIPTION" '[.result.rules[]? | select(.description == $d)][0].id // empty' <<<"$CF_BODY")"
  if [[ -n "$rule_id" ]]; then
    cf PATCH "/zones/$CF_ZONE_ID/rulesets/$ruleset/rules/$rule_id" "$rule"
    cf_ok
    note "updated our rule"
  else
    cf POST "/zones/$CF_ZONE_ID/rulesets/$ruleset/rules" "$rule"
    cf_ok
    note "added our rule (other rules in the phase are untouched)"
  fi
}

# ---- 5. AI Gateway and custom provider -----------------------------------------------------------

step_gateway() {
  say "5. AI Gateway '$GATEWAY_ID' (authenticated; logs=$GATEWAY_COLLECT_LOGS, cache_ttl=$GATEWAY_CACHE_TTL, no gateway rate limit)"
  local path="/accounts/$CF_ACCOUNT_ID/ai-gateway/gateways"
  # Rate limits stay off: Kindling limits each key itself, and a gateway limit is shared by every caller.
  local settings
  settings="$(jq -n --argjson logs "$GATEWAY_COLLECT_LOGS" --argjson ttl "$GATEWAY_CACHE_TTL" \
    '{cache_ttl: $ttl, cache_invalidate_on_update: true, collect_logs: $logs,
      rate_limiting_interval: 0, rate_limiting_limit: 0, authentication: true}')"
  if [[ $APPLY -eq 0 ]]; then
    plan GET "$path/$GATEWAY_ID"
    note "if it doesn't exist:"
    plan POST "$path" "$(jq --arg id "$GATEWAY_ID" '{id: $id} + .' <<<"$settings")"
    note "else bring its settings in line:"
    plan PUT "$path/$GATEWAY_ID" "$settings"
    return
  fi
  cf GET "$path/$GATEWAY_ID"
  if [[ "$CF_STATUS" == 404 ]]; then
    cf POST "$path" "$(jq --arg id "$GATEWAY_ID" '{id: $id} + .' <<<"$settings")"
    cf_ok
    note "created"
  else
    cf_ok
    cf PUT "$path/$GATEWAY_ID" "$settings"
    cf_ok
    note "updated"
  fi
}

step_provider() {
  say "6. Custom provider 'custom-$PROVIDER_SLUG' -> https://$KINDLING_HOSTNAME (account-wide, used by every gateway)"
  local path="/accounts/$CF_ACCOUNT_ID/ai-gateway/custom-providers"
  local body
  body="$(jq -n --arg slug "$PROVIDER_SLUG" --arg url "https://$KINDLING_HOSTNAME" \
    '{name: "Kindling", slug: $slug, base_url: $url, enable: true,
      description: "Self-hosted Nemotron through Kindling (Cloudflare Tunnel)"}')"
  if [[ $APPLY -eq 0 ]]; then
    plan GET "$path?search=$PROVIDER_SLUG"
    note "if there is none with that slug:"
    plan POST "$path" "$body"
    note "else:"
    plan PATCH "$path/<provider-id>" "$(jq 'del(.slug)' <<<"$body")"
    return
  fi
  cf GET "$path?search=$PROVIDER_SLUG"
  cf_ok
  local id
  id="$(jq -r --arg s "$PROVIDER_SLUG" '[.result[]? | select(.slug == $s)][0].id // empty' <<<"$CF_BODY")"
  if [[ -n "$id" ]]; then
    cf PATCH "$path/$id" "$(jq 'del(.slug)' <<<"$body")"
    cf_ok
    note "updated ($id)"
  else
    cf POST "$path" "$body"
    cf_ok
    note "created"
  fi
}

# ---- run -----------------------------------------------------------------------------------------

if [[ $DO_TUNNEL -eq 1 ]]; then
  step_tunnel
  step_config
  step_dns
fi
step_origin_rule
if [[ $DO_GATEWAY -eq 1 ]]; then
  step_gateway
  step_provider
fi

GATEWAY_BASE="https://gateway.ai.cloudflare.com/v1/$CF_ACCOUNT_ID/$GATEWAY_ID/custom-$PROVIDER_SLUG"
say "Next"
cat <<EOF
    Start the tunnel (foreground, to watch it connect):
        cloudflared tunnel --config $KINDLING_CF_DIR/config.yml run
    Or at login, with launchd:
        cp $KINDLING_CF_DIR/launchd/dev.kindling.cloudflared.plist ~/Library/LaunchAgents/
        launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/dev.kindling.cloudflared.plist
    Check each hop:
        curl -s http://127.0.0.1:8000/api/health
        curl -s https://$KINDLING_HOSTNAME/api/health
        curl -s https://$KINDLING_HOSTNAME/v1/models -H "Authorization: Bearer \$KINDLING_KEY"
        curl -s $GATEWAY_BASE/v1/models \\
             -H "cf-aig-authorization: Bearer \$CF_AIG_TOKEN" -H "Authorization: Bearer \$KINDLING_KEY"
    Gateway base URL for OpenAI SDKs: $GATEWAY_BASE/v1
    Create the gateway token (AI Gateway: Run) in the dashboard; see docs/cloudflare.md.
EOF
[[ -n "$ORIGIN_SECRET" ]] && echo "    Then set KINDLING_ORIGIN_SECRET in Kindling's .env and restart it."
[[ $APPLY -eq 0 ]] && echo "
DRY RUN finished: nothing was changed."
exit 0
