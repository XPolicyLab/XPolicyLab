#!/usr/bin/env bash
# One episode: reset, run the agent in its container, collect the result. robo-server must be running.
#   run_episode.sh TASK LAYOUT [OUT_ROOT]
# Environment: LANE, ROBO_PORT, ADMIN_PORT, MODEL, EFFORT, WALL, MAX_TRIES
set -uo pipefail
TASK=$1; LAYOUT=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_ROOT=${3:-$ROOT/runs/eval}
mkdir -p "$OUT_ROOT"; OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
export LANE=${LANE:-0} ROBO_PORT=${ROBO_PORT:-28700}
ADMIN_PORT=${ADMIN_PORT:-$((ROBO_PORT + 1))}
export ROBO_ADMIN_SERVER=http://127.0.0.1:$ADMIN_PORT
ADMIN="python3 $ROOT/roboshell/client/robo_admin.py"
RETRY='429|Too Many Requests|exceeded retry limit|at capacity|server_is_overloaded|serverOverloaded|rate limit|currently overloaded|stream disconnected|50[234] |Bad Gateway'
gateway_failed() {  # the agent's turn was ended by the model gateway (rate limit, overload, disconnect), not by the agent
  grep -F '"type":"turn.failed"' "$1/codex_events.jsonl" 2>/dev/null | grep -qE "$RETRY"
}

for try in $(seq 1 "${MAX_TRIES:-3}"); do
  RUN=$OUT_ROOT/${TASK}_l${LAYOUT}/attempt_$try
  rm -rf "$RUN"; mkdir -p "$RUN/work"
  INSTRUCTION=$($ADMIN reset --task "$TASK" --layout "$LAYOUT" 2> "$RUN/reset.err"); rc=$?
  if [ $rc -eq 5 ]; then echo "{\"task\": \"$TASK\", \"layout\": $LAYOUT, \"skipped\": \"unstable_layout\"}" > "$OUT_ROOT/${TASK}_l${LAYOUT}/result.json"; echo "unstable layout, skipped"; exit 0; fi
  if [ $rc -ne 0 ]; then echo "reset failed"; cat "$RUN/reset.err"; exit 1; fi
  echo "$INSTRUCTION" > "$RUN/instruction.txt"
  cp "$ROOT/agent/frame.png" "$RUN/work/frame.png"
  python3 "$ROOT/roboshell/client/robo_admin.py" result > /dev/null
  MANUAL=$(python3 "$ROOT/roboshell/docs.py" build --task "$TASK")
  AGENTS_MD=$MANUAL WALL=${WALL:-3600} "$ROOT/agent/codex/run_agent.sh" "$RUN" "$INSTRUCTION" > "$RUN/run_agent.out" 2>&1
  $ADMIN result > "$RUN/result.json"
  if [ "$(python3 -c "import json;print(json.load(open('$RUN/result.json')).get('over'))")" != "True" ]; then
    if $ADMIN finish > "$RUN/result.finish.json" 2>/dev/null && [ -s "$RUN/result.finish.json" ]; then
      mv "$RUN/result.finish.json" "$RUN/result.json"   # the agent stopped without done: judged and closed now
    fi
  fi
  python3 "$ROOT/eval/audit.py" "$RUN" > /dev/null 2>&1
  OVER=$(python3 -c "import json;print(json.load(open('$RUN/result.json')).get('over'))")
  if [ ! -s "$RUN/codex_events.jsonl" ]; then
    echo "attempt $try: the agent did not start"; sed -n 1,3p "$RUN/run_agent.out"; sleep 10; continue
  fi
  SUCCESS=$(python3 -c "import json;print(json.load(open('$RUN/result.json')).get('success_official'))")
  if [ "$SUCCESS" != "True" ] && gateway_failed "$RUN"; then   # an episode cut short by the gateway is not a result
    echo "attempt $try: model gateway failure ($(grep -oE "$RETRY" "$RUN/codex_events.jsonl" | head -1)), waiting $((120 * try)) s and retrying"; sleep $((120 * try)); continue
  fi
  cp "$RUN/result.json" "$OUT_ROOT/${TASK}_l${LAYOUT}/result.json"
  echo "$try" > "$OUT_ROOT/${TASK}_l${LAYOUT}/accepted_attempt"
  cat "$RUN/run_agent.out" | tail -1
  python3 -c "import json;r=json.load(open('$RUN/result.json'));print({k:r.get(k) for k in ('over','end_reason','success_official','success_at_done','progress_score','commands','sim_time_s')})"
  exit 0
done
REASON=infra; gateway_failed "$RUN" && REASON=gateway
echo "{\"task\": \"$TASK\", \"layout\": $LAYOUT, \"skipped\": \"$REASON\"}" > "$OUT_ROOT/${TASK}_l${LAYOUT}/result.json"
echo "gave up after ${MAX_TRIES:-3} attempts"; exit 2
