#!/bin/bash
set -e

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
policy_gpu_id=${6}
seed=${7}
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

# Auto-detect model_path if not provided
RUN_BASENAME="${task_name}-go1-${action_type}-${expert_data_num}eps-seed${seed}"
RUNNAME="${RUNNAME:-${RUN_BASENAME}}"
RUN_DIR="${SCRIPT_DIR}/checkpoints/${RUNNAME}"
LATEST_FILE="${SCRIPT_DIR}/checkpoints/${RUN_BASENAME}.latest"

if [ -z "${MODEL_PATH}" ]; then
    if [ -f "${LATEST_FILE}" ]; then
        RUN_DIR="$(cat "${LATEST_FILE}")"
    elif [ ! -d "${RUN_DIR}" ]; then
        RUN_DIR=$(ls -dt "${SCRIPT_DIR}/checkpoints/${RUN_BASENAME}"-* 2>/dev/null | head -1)
    fi
    if [ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ]; then
        # Find the latest checkpoint-N subdirectory
        MODEL_PATH=$(ls -d "${RUN_DIR}"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
    fi
    if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then
        echo -e "\033[31m[ERROR] No checkpoint found in ${RUN_DIR}\033[0m"
        exit 1
    fi
    echo -e "\033[33m[INFO] Auto-detected model_path: ${MODEL_PATH}\033[0m"
fi

# Auto-detect data_stats_path (lives in the run dir, not the checkpoint subdir)
if [ -n "${MODEL_PATH}" ] && [[ "$(basename "${MODEL_PATH}")" == checkpoint-* ]]; then
    RUN_DIR="$(dirname "${MODEL_PATH}")"
fi
DATA_STATS_PATH="${RUN_DIR}/dataset_stats.json"
if [ ! -f "${DATA_STATS_PATH}" ]; then
    DATA_STATS_PATH="${MODEL_PATH}/dataset_stats.json"
fi
if [ ! -f "${DATA_STATS_PATH}" ]; then
    echo -e "\033[31m[ERROR] dataset_stats.json not found. Checked: ${RUN_DIR}/dataset_stats.json and ${MODEL_PATH}/dataset_stats.json\033[0m"
    exit 1
else
    echo -e "\033[33m[INFO] Using data_stats_path: ${DATA_STATS_PATH}\033[0m"
fi

additional_info="model_path=${MODEL_PATH},data_stats_path=${DATA_STATS_PATH},action_type=${action_type}"

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
    "${MODEL_PATH}" \
    "${DATA_STATS_PATH}" &

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
