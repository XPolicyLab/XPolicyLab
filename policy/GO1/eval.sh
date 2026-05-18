#!/bin/bash
set -e

# ==================== 参数定义 ====================
policy_name=GO1
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
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")

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

# 定义 cleanup 函数以确保脚本退出时能正确清理后台进程
cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# ==================== 启动 server ====================
echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# Add AgiBot-World to PYTHONPATH for GO1 model imports
export PYTHONPATH="${SCRIPT_DIR}/AgiBot-World:${PYTHONPATH}"

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
        data_stats_path="${DATA_STATS_PATH}" \
    &
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

bash "${UTILS_DIR}/setup_env_client.sh" "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" "${FREE_PORT}" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${ROOT_DIR}"
echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"