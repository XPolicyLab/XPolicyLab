#!/bin/bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> joint <seed> <gpu_id> [options]" >&2
    exit 1
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${WORKSPACE_ROOT}" "${env_cfg_type}")
exec env CUDA_VISIBLE_DEVICES="${gpu_id}" python "${SCRIPT_DIR}/train.py" \
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    --action-dim "${action_dim}" "$@"
