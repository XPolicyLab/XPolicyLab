#!/bin/bash
set -e

policy_name=A1
dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
gpu_id=${6}
seed=${7}
policy_conda_env=${8}
eval_env_conda_env=${9}
MODEL_PATH=${10:-""}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A1_DIR="${SCRIPT_DIR}/A1"
yaml_file="${SCRIPT_DIR}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")

RUN_BASENAME="${task_name}-a1-${action_type}-${expert_data_num}eps-seed${seed}"
RUN_DIR="${SCRIPT_DIR}/checkpoints/${RUN_BASENAME}"
LATEST_FILE="${SCRIPT_DIR}/checkpoints/${RUN_BASENAME}.latest"

if [ -z "${MODEL_PATH}" ]; then
    if [ -f "${LATEST_FILE}" ]; then
        RUN_DIR="$(cat "${LATEST_FILE}")"
    elif [ ! -d "${RUN_DIR}" ]; then
        RUN_DIR=$(ls -dt "${SCRIPT_DIR}/checkpoints/${RUN_BASENAME}"-* 2>/dev/null | head -1 || true)
    fi
    if [ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ]; then
        MODEL_PATH=$(find "${RUN_DIR}" -maxdepth 1 -type d -name '*-unsharded' | sort | tail -1)
    fi
    if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then
        MODEL_PATH="/mnt/pfs/pg4hw0/qiwei/models/a1-pretrain"
    fi
fi
echo -e "\033[33m[INFO] Using model_path: ${MODEL_PATH}\033[0m"
if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "\033[31m[ERROR] model_path does not exist: ${MODEL_PATH}\033[0m"
    exit 1
fi
if [ ! -f "${MODEL_PATH}/model.pt" ]; then
    echo -e "\033[31m[ERROR] model_path is missing model.pt: ${MODEL_PATH}\033[0m"
    exit 1
fi
if [ ! -f "${MODEL_PATH}/config.yaml" ] && [ ! -f "$(dirname "${MODEL_PATH}")/config.yaml" ]; then
    echo -e "\033[31m[ERROR] config.yaml not found in model_path or parent: ${MODEL_PATH}\033[0m"
    exit 1
fi
if [[ "${MODEL_PATH}" == *"-joint-"* && "${action_type}" != "joint" ]]; then
    echo -e "\033[31m[ERROR] model_path is a joint checkpoint, but action_type='${action_type}'. Use action_type='joint'.\033[0m"
    exit 1
fi
if [[ "${MODEL_PATH}" == *"-ee-"* && "${action_type}" != "ee" ]]; then
    echo -e "\033[31m[ERROR] model_path is an ee checkpoint, but action_type='${action_type}'. Use action_type='ee'.\033[0m"
    exit 1
fi

cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export PYTHONPATH="${A1_DIR}:${PYTHONPATH}"
export DATA_DIR="${DATA_DIR:-/mnt/pfs/pg4hw0/qiwei/models}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        env_cfg_type="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        model_path="${MODEL_PATH}" \
    &
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

echo -e "\033[32m[SERVER] Waiting for model server to listen on port ${FREE_PORT}...\033[0m"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
start_ts=$(date +%s)
while true; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo -e "\033[31m[ERROR] Server process exited before opening port ${FREE_PORT}.\033[0m"
        wait "${SERVER_PID}" || true
        exit 1
    fi
    if python - "${FREE_PORT}" <<'PY'
import socket
import sys
port = int(sys.argv[1])
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
PY
    then
        echo -e "\033[32m[SERVER] Port ${FREE_PORT} is ready.\033[0m"
        break
    fi
    now_ts=$(date +%s)
    if [ $((now_ts - start_ts)) -ge "${SERVER_WAIT_TIMEOUT}" ]; then
        echo -e "\033[31m[ERROR] Timeout waiting for server port ${FREE_PORT} after ${SERVER_WAIT_TIMEOUT}s.\033[0m"
        exit 1
    fi
    sleep 2
done

bash "${UTILS_DIR}/setup_env_client.sh" "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" "${FREE_PORT}" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${ROOT_DIR}"
echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"
