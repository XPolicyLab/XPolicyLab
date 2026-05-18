#!/bin/bash
set -e

policy_name=X_VLA
dataset_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
expert_data_num=${5}
action_type=${6}
seed=${7}
policy_gpu_id=${8}
env_gpu_id=${9}
policy_conda_env=${10}
eval_env_conda_env=${11}

MODEL_PATH=${10}
PROCESSOR_PATH=${11}

export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${policy_gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/X-VLA/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")

cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        env_cfg="${env_cfg}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        model_path="${MODEL_PATH}" \
        processor_path="${PROCESSOR_PATH}" \
    &
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

# ==================== 启动 client 进行评测 ====================
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
bash "${UTILS_DIR}/setup_env_client.sh" "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" "${FREE_PORT}" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${additional_info}" "${ROOT_DIR}" "${seed}" "${env_gpu_id}"
echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"
