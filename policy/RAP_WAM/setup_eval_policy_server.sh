#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 ]]; then echo 'Expected standard server arguments' >&2; exit 2; fi
bench=$1; task=$2; ckpt=$3; robot=$4; action=$5; seed=$6; gpu=$7; policy_env=$8; port=$9; host=${10:-localhost}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${DIR}/../.." && pwd)"
source "$DIR/runtime_env.sh"
export PYTHONPATH="${XPL_ROOT}:$(dirname "${XPL_ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$gpu"
py="$(rap_wam_python_for_env "$policy_env")"
cd "$RAP_WAM_WORK_DIR/run"
exec "$py" "${XPL_ROOT}/setup_policy_server.py" --config_path "${DIR}/deploy.yml" --overrides \
  "port=$port" "host=$host" "bench_name=$bench" "task_name=$task" "ckpt_name=$ckpt" \
  "env_cfg_type=$robot" "seed=$seed" "policy_name=RAP_WAM" "action_type=$action"
