#!/usr/bin/env bash
# Start robo-server for one task. Usage: start_server.sh GPU TASK [SEED]
# Environment: LANE (default 0), PORT (default 28700), ADMIN_PORT (PORT+1), RUN_ROOT, MIN_FREE_MIB
set -euo pipefail
GPU=$1; TASK=$2; SEED=${3:-0}
source "$(dirname "$0")/sim_env.sh"
export LANE=${LANE:-0}
PORT=${PORT:-28700}; ADMIN_PORT=${ADMIN_PORT:-$((PORT + 1))}
for p in $PORT $ADMIN_PORT; do
  if ss -ltn | grep -qE ":$p\b"; then echo "port $p is in use" >&2; exit 1; fi
done
HOST=${HOST:-$("$ROBOSHELL_ROOT/agent/codex/run_agent.sh" --gateway)}
RUN_ROOT=${RUN_ROOT:-$ROBOSHELL_ROOT/runs/episodes/$TASK}
mkdir -p "$RUN_ROOT"
inner() {
  # one visible GPU only: cuRobo always uses cuda:0, so the simulator must be on cuda:0 too
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ROBODOJO_REPO"
  exec "$ISAAC_PYTHON" -u "$ROBOSHELL_ROOT/roboshell/server/main.py" \
    --task "$TASK" --seed "$SEED" --device_id "$GPU" --device cuda:0 \
    --host "$HOST" --port "$PORT" --admin-port "$ADMIN_PORT" --run-root "$RUN_ROOT" \
    --headless --enable_cameras --kit_args "$KIT_ARGS"
}
export -f inner; export GPU TASK SEED HOST PORT ADMIN_PORT RUN_ROOT KIT_ARGS
echo "robo-server: task=$TASK seed=$SEED gpu=$GPU agent=$HOST:$PORT admin=127.0.0.1:$ADMIN_PORT log=$RUN_ROOT/server.log"
run_on_gpu "$GPU" "RoboShell-server-$TASK" "${MIN_FREE_MIB:-12000}" bash -c inner >> "$RUN_ROOT/server.log" 2>&1
