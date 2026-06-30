#!/bin/bash
set -e
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
env_gpu_id=$9
policy_uv_env=${10:-uv}
eval_env_conda_env=${11}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Current Dir
ROOT_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

SERVER_SCRIPT="${CURRENT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${CURRENT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill -TERM -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

setsid bash "${SERVER_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_uv_env}" \
    "${policy_server_port}" \
    "${policy_server_ip}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"
deactivate
bash "${CLIENT_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo "[MAIN] eval finished"