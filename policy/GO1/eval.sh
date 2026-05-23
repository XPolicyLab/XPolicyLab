#!/bin/bash
set -e

dataset_name=$1
task_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
policy_gpu_id=$6
seed=$7
default_conda_env="${CONDA_DEFAULT_ENV:-}"
policy_conda_env=${8:-${default_conda_env}}
eval_env_conda_env=${9:-${policy_conda_env}}
MODEL_PATH=${10:-""}
env_gpu_id=${11:-${policy_gpu_id}}

if [[ -z "${policy_conda_env}" ]]; then
    echo -e "\033[31m[ERROR] policy_conda_env is empty. Pass it explicitly or run inside an activated conda env.\033[0m"
    exit 1
fi

if [[ -z "${eval_env_conda_env}" ]]; then
    echo -e "\033[31m[ERROR] eval_env_conda_env is empty. Pass it explicitly or run inside an activated conda env.\033[0m"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${MODEL_PATH}" &

SERVER_PID=$!

sleep 3

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo "[MAIN] eval finished"
