#!/usr/bin/env bash
set -euo pipefail
BENCH=${1:?} TASK=${2:?} CHECKPOINT=${3:?} ROBOT=${4:?} ACTION=${5:?} SEED=${6:?}
GPU=${7:?} POLICY_ENV=${8:?} PORT=${9:?} HOST=${10:-127.0.0.1}
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$POLICY_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
if [[ -x "$POLICY_ENV/bin/python" ]]; then
  POLICY_PYTHON="$POLICY_ENV/bin/python"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$POLICY_ENV"
  POLICY_PYTHON="$(command -v python)"
fi
export PYTHONPATH="$BENCH_ROOT:$XPL_ROOT:${PYTHONPATH:-}"
exec env CUDA_VISIBLE_DEVICES="$GPU" "$POLICY_PYTHON" "$XPL_ROOT/setup_policy_server.py" \
  --config_path "$POLICY_DIR/deploy.yml" --overrides "host=$HOST" "port=$PORT" \
  "ckpt_name=$CHECKPOINT" "env_cfg_type=$ROBOT" "action_type=$ACTION" "seed=$SEED" \
  "bench_name=$BENCH" "task_name=$TASK"
