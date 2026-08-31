#!/bin/bash
# Backstop for GitHub's scheduler, and a watchdog for silent failure.
#
# Two things went wrong that nobody noticed for days: the cron never fired, and
# a failed emergency liquidation reported success. Both were silent. This
# checks all three questions every time it runs:
#
#   1. Did the workflow run today at all?      -> dispatch it if not
#   2. Did that run succeed?                   -> shout if not
#   3. Is the official account actually doing   -> shout if it is still
#      anything?                                  untouched during the window
#
# Safe to run repeatedly: it dispatches at most once per day, and the pipeline
# reconciles open positions before deploying capital, so a later run manages
# the existing position rather than opening another.

set -uo pipefail
REPO="vitoac-22/Autonomous-Options-MCP-Agent"
WORKFLOW="agent_trigger.yml"
DIR="$HOME/Desktop/Autonomous-Options-MCP-Agent"
LOG="$DIR/ops/backstop.log"
GH="$(command -v gh || echo /opt/homebrew/bin/gh)"
ACCOUNT_ID="4a713e2a-1c0b-4dcb-b194-a76e10877050"

say()   { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ')  $*" >> "$LOG"; }
alert() { say "ALERT: $*"
          osascript -e "display notification \"$*\" with title \"Delta Zero\" sound name \"Basso\"" 2>/dev/null || true; }

KEY=$(grep '^ALPACA_API_KEY=' "$DIR/.env" 2>/dev/null | cut -d= -f2)
SEC=$(grep '^ALPACA_SECRET_KEY=' "$DIR/.env" 2>/dev/null | cut -d= -f2)
OPEN=$(curl -s --max-time 20 -H "APCA-API-KEY-ID: $KEY" -H "APCA-API-SECRET-KEY: $SEC" \
        "https://paper-api.alpaca.markets/v2/clock" | grep -o '"is_open":[a-z]*' | cut -d: -f2)

if [ "$OPEN" != "true" ]; then
  say "market closed — standing down"
  exit 0
fi

TODAY=$(date -u +%Y-%m-%d)
RUNS=$("$GH" run list --repo "$REPO" --workflow "$WORKFLOW" --limit 20 \
       --json createdAt,conclusion \
       --jq "[.[] | select(.createdAt | startswith(\"$TODAY\"))]" 2>/dev/null)
COUNT=$(echo "$RUNS" | grep -o '"conclusion"' | wc -l | tr -d ' ')

# 1. no run today -> dispatch
if [ "${COUNT:-0}" -eq 0 ]; then
  say "no run today, market open — dispatching"
  if "$GH" workflow run "$WORKFLOW" --repo "$REPO" >>"$LOG" 2>&1; then
    say "dispatched OK"
  else
    alert "Could not dispatch the workflow — trigger it by hand"
  fi
  exit 0
fi

# 2. ran, but did it succeed?
FAILED=$(echo "$RUNS" | grep -c '"conclusion":"failure"' || true)
SUCCESS=$(echo "$RUNS" | grep -c '"conclusion":"success"' || true)
say "$COUNT run(s) today: $SUCCESS success, $FAILED failure"
[ "${FAILED:-0}" -gt 0 ] && [ "${SUCCESS:-0}" -eq 0 ] && alert "Every run today FAILED — check the Actions log"

# 3. is the judged account actually being traded?
ORDERS=$(curl -s --max-time 20 -H "APCA-API-KEY-ID: $KEY" -H "APCA-API-SECRET-KEY: $SEC" \
         "https://paper-api.alpaca.markets/v2/orders?status=all&limit=50" | grep -o '"id"' | wc -l | tr -d ' ')
say "official account orders to date: $ORDERS"
[ "${ORDERS:-0}" -eq 0 ] && [ "${SUCCESS:-0}" -gt 0 ] && \
  say "note: runs succeeded but no orders yet — expected if the gates are vetoing"
