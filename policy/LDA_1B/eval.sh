#!/bin/bash
# Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
#            <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
env_gpu_id=${8:?env_gpu_id required}
policy_conda_env=${9:?policy_conda_env required}
eval_env_conda_env=${10:?eval_env_conda_env required}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo -e "\033[32m[MAIN] start server, policy_server_port=${policy_server_port}\033[0m"

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
    "${policy_server_ip}" &

SERVER_PID=$!
sleep 3

echo -e "\033[32m[MAIN] start client, server=${policy_server_ip}:${policy_server_port}\033[0m"

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

echo -e "\033[33m[MAIN] eval finished\033[0m"
