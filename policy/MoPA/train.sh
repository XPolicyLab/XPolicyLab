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
if [[ "${env_cfg_type,,}" == "m92uw" || "${env_cfg_type,,}" == "mobile" || "${env_cfg_type,,}" == "mobile_m92uw" ]]; then
    action_dim=75
else
    action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${WORKSPACE_ROOT}" "${env_cfg_type}")
fi
if [[ "${gpu_id}" == "all" || "${gpu_id}" == *,* ]]; then
    if [[ "${gpu_id}" == "all" ]]; then
        gpu_id="$(python -c 'import torch; print(",".join(map(str, range(torch.cuda.device_count()))))')"
        if [[ -z "${gpu_id}" ]]; then
            echo "No CUDA devices found for gpu_id=all." >&2
            exit 1
        fi
    fi
    IFS=',' read -r -a gpu_list <<< "${gpu_id}"
    nproc=${#gpu_list[@]}
    exec env CUDA_VISIBLE_DEVICES="${gpu_id}" torchrun --standalone --nproc_per_node="${nproc}" \
        "${SCRIPT_DIR}/train.py" "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
        --action-dim "${action_dim}" "$@"
fi
exec env CUDA_VISIBLE_DEVICES="${gpu_id}" python "${SCRIPT_DIR}/train.py" \
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    --action-dim "${action_dim}" "$@"
