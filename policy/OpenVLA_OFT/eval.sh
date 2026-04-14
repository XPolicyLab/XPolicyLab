#!/bin/bash
set -e

# ==================== 参数定义 ====================
policy_name=openvla-oft
task_name=${1}
env_cfg=${2}
expert_data_num=${3}
action_type=${4}
gpu_id=${5}
seed=${6}
policy_conda_env=${7}
eval_env_conda_env=${8}
CHECKPOINT_PATH=${9}
UNNORM_KEYS=${10}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")

# 定义 cleanup 函数以确保脚本退出时能正确清理后台进程
cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# ==================== 启动 server ====================
echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        task_name="${task_name}" \
        env_cfg="${env_cfg}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        checkpoint_path="${CHECKPOINT_PATH}" \
        unnorm_key=${UNNORM_KEYS} \
    &

SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

# ==================== 启动 client 进行评测 ====================
bash "${UTILS_DIR}/run_debug_env_client.sh" "1" "${eval_env_conda_env}" "${FREE_PORT}" "${task_name}" "${env_cfg}" "${policy_name}" "${ROOT_DIR}"
echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"