#!/bin/bash
set -e

# ==================== 参数定义 ====================
policy_name=Pi_05
task_name=${1}
env_cfg_type=${2}
expert_data_num=${3}
action_type=${4}
gpu_id=${5}
seed=${6}
policy_uv_env_path=${7}
eval_env_conda_env=${8}
MODEL_PATH=${9}
TRAIN_CONFIG_NAME=${10}
REPO_ID=${11}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")

# 定义 cleanup 函数以确保脚本退出时能正确清理后台进程
cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# ==================== 启动 server ====================
echo -e "\033[32m[SERVER] Activating uv environment: ${policy_uv_env_path}\033[0m"

source ${policy_uv_env_path}/.venv/bin/activate

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        task_name="${task_name}" \
        env_cfg_type="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        model_path="${MODEL_PATH}" \
        train_config_name="${TRAIN_CONFIG_NAME}" \
        repo_id="${REPO_ID}" \
&
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

# ==================== 启动 client 进行评测 ====================
bash "${UTILS_DIR}/run_debug_env_client.sh" false "${eval_env_conda_env}" "${FREE_PORT}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${ROOT_DIR}"
echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"