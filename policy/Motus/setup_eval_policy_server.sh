#!/bin/bash
set -e

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_conda_env=$9
port=${10}
host=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
UTILS_DIR="${ROOT_DIR}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ENV_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${port}, action_dim=${action_dim}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${ROOT_DIR}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            host="${host}" \
            port="${port}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
