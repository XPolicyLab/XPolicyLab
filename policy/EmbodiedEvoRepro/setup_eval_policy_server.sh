#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 9 ]] || { echo 'Usage: setup_eval_policy_server.sh bench task ckpt robot action seed gpu python_env port [host]' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/shell_common.sh"
validate_mode
activate_python "$8"
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# The code agent is isolated upstream without GPU access; this is the policy host.
exec env CUDA_VISIBLE_DEVICES="$7" python "${SCRIPT_DIR}/serve.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" --overrides \
    "bench_name=$1" "task_name=$2" "ckpt_name=$3" "env_cfg_type=$4" \
    "action_type=$5" "seed=$6" "gpu_id=$7" "port=$9" "host=${10:-localhost}"
