#!/bin/bash
set -euo pipefail

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=${9}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"
ADAPTER_DIR="${SCRIPT_DIR}/LDA-1B/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"

expert_data_num="${LDA_EXPERT_DATA_NUM:-}"

if [[ -n "${LDA_CHECKPOINT_PATH:-}" ]]; then
    checkpoint_path="${LDA_CHECKPOINT_PATH}"
elif ! checkpoint_path="$(xpolicylab_resolve_checkpoint_pt "${SCRIPT_DIR}" "${dataset_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}")"; then
    ckpt_run_id="$(xpolicylab_ckpt_run_id "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
    echo -e "\033[31m[SERVER] checkpoint not found for ckpt_run_id=${ckpt_run_id}\033[0m" >&2
    echo -e "\033[31m[SERVER] (eval args: dataset=${dataset_name} ckpt_name=${ckpt_name} env=${env_cfg_type} action=${action_type} seed=${seed})\033[0m" >&2
    echo -e "\033[31m[SERVER] Set LDA_CHECKPOINT_PATH=... or LDA_EXPERT_DATA_NUM=... for legacy layouts.\033[0m" >&2
    exit 1
fi

echo -e "\033[33m[SERVER] GPU=${policy_gpu_id} host=${policy_server_host} port=${policy_server_port}\033[0m"
echo -e "\033[33m[SERVER] task_name=${task_name} ckpt_name=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] checkpoint_path=${checkpoint_path}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
cd "${SCRIPT_DIR}/LDA-1B"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="LDA_1B" \
            action_type="${action_type}" \
            checkpoint_path="${checkpoint_path}"
