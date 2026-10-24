#!/usr/bin/env bash
set -euo pipefail
BENCH=${1:?} TASK=${2:?} CHECKPOINT=${3:?} ROBOT=${4:?} ACTION=${5:?} SEED=${6:?}
GPU=${7:?} ENV_GPU=${8:?} POLICY_ENV=${9:?} EVAL_ENV=${10:?}
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$POLICY_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
if [[ "${EVAL_ENV_TYPE:-debug}" != debug ]]; then
  echo 'Real robot evaluation uses the official remote client; start the policy server with setup_eval_policy_server.sh.' >&2
  exit 2
fi
if [[ -x "$POLICY_ENV/bin/python" ]]; then
  POLICY_PYTHON="$POLICY_ENV/bin/python"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$POLICY_ENV"
  POLICY_PYTHON="$(command -v python)"
fi
export PYTHONPATH="$BENCH_ROOT:$XPL_ROOT:${PYTHONPATH:-}"
PORT=$(bash "$XPL_ROOT/utils/get_free_port.sh")
SERVER_PID=''
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
CUDA_VISIBLE_DEVICES="$GPU" "$POLICY_PYTHON" "$XPL_ROOT/setup_policy_server.py" \
  --config_path "$POLICY_DIR/deploy.yml" --overrides \
  "host=127.0.0.1" "port=$PORT" "ckpt_name=$CHECKPOINT" "env_cfg_type=$ROBOT" \
  "action_type=$ACTION" "seed=$SEED" "bench_name=$BENCH" "task_name=$TASK" &
SERVER_PID=$!
bash "$XPL_ROOT/utils/wait_for_policy_server.sh" 127.0.0.1 "$PORT" "$SERVER_PID" 'Piper X policy' 600
"$POLICY_PYTHON" "$XPL_ROOT/utils/debug_env_client.py" --bench_name "$BENCH" --task_name "$TASK" \
  --env_cfg_type "$ROBOT" --policy_name JAM --protocol ws --host 127.0.0.1 --port "$PORT" \
  --eval_episode_num 1 --evaluation_id piperx-interface-debug
echo 'PIPERX_WEBSOCKET_DEBUG_PASSED (interface only; no physical rollout)'
echo '[MAIN] eval finished'
