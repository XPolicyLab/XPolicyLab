#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 ]]; then
    echo "Usage: $0 <bench> <task> <ckpt> <env_cfg> <action> <seed> <gpu> <uv_or_project> <port> [host]" >&2
    exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
sla_python "$8"
export CUDA_VISIBLE_DEVICES="$7"
cd "${SLA_DIR}"
exec "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SLA_DIR}/deploy.yml" --overrides \
    policy_name=SLA "bench_name=$1" "task_name=$2" "ckpt_name=$3" \
    "env_cfg_type=$4" "action_type=$5" "seed=$6" "gpu_id=$7" \
    "port=$9" "host=${10:-localhost}"
