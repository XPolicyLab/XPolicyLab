#!/bin/bash
set -e
set -o pipefail

bench_name=${1:?bench_name is required}
task_name=${2:?task_name is required}
ckpt_name=${3:?ckpt_name is required}
env_cfg_type=${4:?env_cfg_type is required}
expert_data_num=${5:?expert_data_num is required}
action_type=${6:?action_type is required}
seed=${7:?seed is required}
policy_gpu_id=${8:?policy_gpu_id is required}
env_gpu_id=${9:?env_gpu_id is required}
default_conda_env="${CONDA_DEFAULT_ENV:-}"
policy_conda_env=${10:-${default_conda_env}}
eval_env_conda_env=${11:-${policy_conda_env}}
model_path=${12:-${MODEL_PATH:-""}}

if [[ -z "${policy_conda_env}" ]]; then
    echo "[ERROR] policy_conda_env is empty. Pass it explicitly or activate the DreamZero conda env."
    exit 1
fi

if [[ -z "${eval_env_conda_env}" ]]; then
    echo "[ERROR] eval_env_conda_env is empty. Pass it explicitly or activate the eval conda env."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
policy_server_host="${DREAMZERO_POLICY_SERVER_HOST:-localhost}"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${policy_server_host}" \
    "${model_path}" &

SERVER_PID=$!

sleep 3

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
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
