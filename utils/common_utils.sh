#!/bin/bash
set -e

# ==================== 参数定义 ====================
task_name=${1}
env_cfg=${2}
expert_data_num=${3}
action_type=${4}
gpu_id=${5}
seed=${6}
policy_conda_env=${7}
sim_conda_env=${8}

if [ -z "${POLICY_NAME}" ]; then
    echo "[ERROR] POLICY_NAME is not set."
    exit 1
fi

if [ -z "${YAML_FILE}" ]; then
    echo "[ERROR] YAML_FILE is not set."
    exit 1
fi

if ! declare -f build_policy_overrides >/dev/null; then
    echo "[ERROR] build_policy_overrides function is not defined."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/common_utils.sh"

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

action_dim=$(get_action_dim "${env_cfg}")
echo -e "\033[33m[INFO] action_dim: ${action_dim}\033[0m"

FREE_PORT=$(get_free_port)
echo -e "\033[33m[INFO] Using socket port: ${FREE_PORT}\033[0m"

echo -e "\033[33m[INFO] Using config file: ${YAML_FILE}\033[0m"

# 拼接公共 overrides
COMMON_OVERRIDES=(
    port="${FREE_PORT}"
    task_name="${task_name}"
    env_cfg="${env_cfg}"
    expert_data_num="${expert_data_num}"
    seed="${seed}"
    policy_name="${POLICY_NAME}"
    action_type="${action_type}"
    action_dim="${action_dim}"
)

# 获取策略私有 overrides
POLICY_OVERRIDES=()
build_policy_overrides POLICY_OVERRIDES \
    "${task_name}" \
    "${env_cfg}" \
    "${expert_data_num}" \
    "${action_type}" \
    "${gpu_id}" \
    "${seed}"

# ==================== 启动 server ====================
echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
activate_conda "${policy_conda_env}"

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python XPolicyLab/setup_policy_server.py \
    --config_path "${YAML_FILE}" \
    --overrides \
    "${COMMON_OVERRIDES[@]}" \
    "${POLICY_OVERRIDES[@]}" \
    &
SERVER_PID=$!

echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

trap 'echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m"; kill ${SERVER_PID} 2>/dev/null || true' EXIT

# ==================== 启动 client ====================
conda deactivate || true
activate_conda "${sim_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${sim_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server port ${FREE_PORT}...\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python XPolicyLab/debug_policy_env.py \
    --task_name "${task_name}" \
    --env_cfg "${env_cfg}" \
    --policy_name "${POLICY_NAME}" \
    --port "${FREE_PORT}"

echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"