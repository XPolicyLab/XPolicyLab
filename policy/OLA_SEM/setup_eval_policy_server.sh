#!/bin/bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
policy_gpu_id=${7:?}
policy_conda_env=${8:?}
policy_server_port=${9:?}
policy_server_host=${10:-127.0.0.1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
wan_path=${OLA_SEM_WAN_PATH:?Set OLA_SEM_WAN_PATH}
vlm_path=${OLA_SEM_VLM_PATH:?Set OLA_SEM_VLM_PATH}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${WORKSPACE_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="OLA_SEM" \
            action_type="${action_type}" \
            wan_path="${wan_path}" \
            vlm_path="${vlm_path}"
