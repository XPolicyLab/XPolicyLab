#!/usr/bin/env bash
# M0 probe inside the agent container. Usage: run_probe_container.sh NAME PROMPT_FILE PORT [fake_server args...]
set -euo pipefail
NAME=$1; PROMPT=$2; PORT=$3; shift 3
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RUN=$ROOT/runs/m0/$NAME
ADMIN_PORT=$((PORT + 1))
export LANE=${LANE:-m0}
for p in $PORT $ADMIN_PORT; do
  if ss -ltn | grep -qE ":$p\b"; then echo "port $p is in use" >&2; exit 1; fi
done
GW=$("$ROOT/agent/codex/run_agent.sh" --gateway)
rm -rf "$RUN"; mkdir -p "$RUN"
PY=$ROOT/.venv/bin/python; [ -x "$PY" ] || PY=$(command -v python3)
"$PY" "$ROOT/tests/probes/codex/fake_server.py" --host "$GW" --port "$PORT" --admin-port "$ADMIN_PORT" "$@" > "$RUN/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
export ROBO_ADMIN_SERVER=http://127.0.0.1:$ADMIN_PORT
for _ in $(seq 50); do curl -s -o /dev/null "$ROBO_ADMIN_SERVER/admin/result" && break; sleep 0.2; done
python3 "$ROOT/roboshell/client/robo_admin.py" reset --task m0_probe --layout 0 > "$RUN/instruction.txt"
ROBO_PORT=$PORT AGENTS_MD=$ROOT/tests/probes/codex/AGENTS.md WALL=${WALL:-1500} "$ROOT/agent/codex/run_agent.sh" "$RUN" "$(cat "$PROMPT")"
python3 "$ROOT/roboshell/client/robo_admin.py" result > "$RUN/server_result.json"
