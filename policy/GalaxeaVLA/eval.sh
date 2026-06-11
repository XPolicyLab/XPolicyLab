#!/bin/bash
# Usage: bash eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> \
#            <action_type> <seed> <policy_gpu_id> <env_gpu_id> \
#            <policy_uv_env_path> <eval_env_conda_env>
set -euo pipefail

dataset_name=${1:?dataset_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
env_gpu_id=${8:?env_gpu_id required}
policy_uv_env_path=${9:?policy_uv_env_path required}
eval_env_conda_env=${10:?eval_env_conda_env required}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

if [[ "${policy_uv_env_path}" != /* ]]; then
    policy_uv_env_path="${SCRIPT_DIR}/${policy_uv_env_path}"
fi
policy_uv_env_path="$(cd "${policy_uv_env_path}" && pwd)"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="${POLICY_SERVER_HOST:-localhost}"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] kill server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo -e "\033[32m[MAIN] policy_uv_env=${policy_uv_env_path}/.venv\033[0m"
echo -e "\033[32m[MAIN] start server, policy_server_port=${policy_server_port}\033[0m"

bash "${SERVER_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_uv_env_path}" \
    "${policy_server_port}" \
    "${policy_server_ip}" &
SERVER_PID=$!
echo -e "\033[32m[MAIN] server PID=${SERVER_PID}\033[0m"

for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo -e "\033[31m[ERROR] Policy server exited before opening port ${policy_server_port}.\033[0m" >&2
        exit 1
    fi
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${policy_server_ip}', int('${policy_server_port}'))); s.close()" >/dev/null 2>&1; then
        echo -e "\033[32m[MAIN] server ready on ${policy_server_ip}:${policy_server_port}\033[0m"
        break
    fi
    sleep 2
done

echo -e "\033[32m[MAIN] start client, server=${policy_server_ip}:${policy_server_port}\033[0m"
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

echo -e "\033[33m[MAIN] eval finished\033[0m"
