#!/bin/bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
    echo "Usage: bash eval.sh <dataset_name> <task_name> <ckpt_name_or_path> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>"
    echo "Example: DEXORA_CKPT_PATH=/root/crx/Dexora/checkpoints/dexora-1b-posttrain/checkpoint-50000/ema/model.safetensors bash eval.sh RoboDojo stack_bowls dexora-1b-posttrain/checkpoint-50000 arx_x5 joint 0 0 1 dexora dexora"
    exit 1
fi

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_host="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start Dexora_1B server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${policy_server_host}" &

SERVER_PID=$!

sleep 8

echo "[MAIN] start client, server=${policy_server_host}:${policy_server_port}"

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
    "${policy_server_host}"

echo "[MAIN] eval finished"
