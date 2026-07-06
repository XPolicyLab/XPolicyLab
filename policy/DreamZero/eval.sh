#!/bin/bash
set -euo pipefail
set -o pipefail

bench_name=${1:?bench_name is required}
task_name=${2:?task_name is required}
ckpt_name=${3:?ckpt_name is required}
env_cfg_type=${4:?env_cfg_type is required}
action_type=${5:?action_type is required}
seed=${6:?seed is required}
policy_gpu_id=${7:?policy_gpu_id is required}
env_gpu_id=${8:?env_gpu_id is required}
default_conda_env="${CONDA_DEFAULT_ENV:-}"
policy_conda_env=${9:-${default_conda_env}}
eval_env_conda_env=${10:-${policy_conda_env}}
model_path=${11:-${MODEL_PATH:-""}}

if [[ -z "${policy_conda_env}" ]]; then
    echo "[ERROR] policy_conda_env is empty. Pass it explicitly or activate the DreamZero conda env."
    exit 1
fi

if [[ -z "${eval_env_conda_env}" ]]; then
    echo "[ERROR] eval_env_conda_env is empty. Pass it explicitly or activate the eval conda env."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

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
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${policy_server_host}" \
    "${model_path}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

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
