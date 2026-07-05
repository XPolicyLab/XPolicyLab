#!/bin/bash
set -e

policy_name="$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
bench_name=${1}
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

if [[ -z "${bench_name}" || -z "${task_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${expert_data_num}" || -z "${action_type}" || -z "${seed}" || -z "${policy_gpu_id}" || -z "${env_gpu_id}" || -z "${policy_conda_env}" || -z "${eval_env_conda_env}" ]]; then
    echo "Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>"
    exit 1
fi

FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo -e "\033[33m[INFO] Policy GPU ID: ${policy_gpu_id}\033[0m"
echo -e "\033[33m[INFO] Env GPU ID: ${env_gpu_id}\033[0m"
echo -e "\033[32m[MAIN] start server, port=${FREE_PORT}\033[0m"

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
    "${FREE_PORT}" &

SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${FREE_PORT}" "${SERVER_PID}" "Policy server" 600

echo -e "\033[32m[MAIN] start client, server=${policy_server_ip}:${FREE_PORT}\033[0m"

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
    "${FREE_PORT}" \
    "${policy_server_ip}"

echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"
