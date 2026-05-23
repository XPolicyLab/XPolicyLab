#!/bin/bash
set -e

dataset_name=$1
task_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
model_path=${10:-""}
policy_server_host=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
A1_DIR="${SCRIPT_DIR}/A1"
yaml_file="${SCRIPT_DIR}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo "[SERVER] policy=A1, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] model_path=${model_path}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export PYTHONPATH="${A1_DIR}:${PYTHONPATH}"
export DATA_DIR="${DATA_DIR:-$(cd "${ROOT_DIR}/.." && pwd)/models}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            policy_server_host="${policy_server_host}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            policy_name="A1" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            model_path="${model_path}"
