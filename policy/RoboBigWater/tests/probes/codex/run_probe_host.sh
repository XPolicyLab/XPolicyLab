#!/usr/bin/env bash
# M0 probe runner (host, Codex sandbox workspace-write). Usage: run_probe.sh NAME PROMPT_FILE [fake_server args...]
set -euo pipefail
NAME=$1; PROMPT=$2; shift 2
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RUN=$ROOT/runs/m0/$NAME
PORT=${PORT:-18700}; ADMIN_PORT=${ADMIN_PORT:-18701}
MODEL=${MODEL:-gpt-6-astra}; EFFORT=${EFFORT:-medium}
export PATH=$HOME/.local/bin:$PATH

rm -rf "$RUN"; mkdir -p "$RUN/work" "$RUN/bin"
cp "$ROOT/tests/probes/codex/AGENTS.md" "$RUN/work/AGENTS.md"
printf '#!/bin/sh\nexec python3 %s/roboshell/client/robo.py "$@"\n' "$ROOT" > "$RUN/bin/robo"
chmod +x "$RUN/bin/robo"

"$ROOT/.venv/bin/python" "$ROOT/tests/probes/codex/fake_server.py" --port "$PORT" --admin-port "$ADMIN_PORT" "$@" > "$RUN/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$ADMIN_PORT/admin/result" && break; sleep 0.2; done

export ROBO_SERVER=http://127.0.0.1:$PORT ROBO_ADMIN_SERVER=http://127.0.0.1:$ADMIN_PORT
python3 "$ROOT/roboshell/client/robo_admin.py" reset --task m0_probe --layout 0 > "$RUN/instruction.txt"

START=$(date +%s)
set +e
PATH="$RUN/bin:$PATH" timeout "${WALL:-1500}" codex exec \
  --json --ephemeral --skip-git-repo-check \
  -s workspace-write -c sandbox_workspace_write.network_access=true \
  -C "$RUN/work" \
  -m "$MODEL" -c model_reasoning_effort="\"$EFFORT\"" \
  --disable memories --disable multi_agent --disable apps --disable plugins \
  -o "$RUN/last_message.txt" \
  "$(cat "$PROMPT")" > "$RUN/codex_events.jsonl" 2> "$RUN/codex_stderr.log"
RC=$?
set -e
echo "{\"rc\": $RC, \"wall_s\": $(( $(date +%s) - START )), \"model\": \"$MODEL\", \"effort\": \"$EFFORT\", \"codex\": \"$(codex --version)\"}" > "$RUN/run_meta.json"
python3 "$ROOT/roboshell/client/robo_admin.py" result > "$RUN/server_result.json"
cat "$RUN/run_meta.json"
