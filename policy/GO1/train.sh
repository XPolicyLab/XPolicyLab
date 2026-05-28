#!/bin/bash
set -e
set -o pipefail

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
gpu_id=$8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
AGIBOT_DIR="${SCRIPT_DIR}/AgiBot-World"
DEFAULT_GO1_MODEL_PATH="$(cd "${ROOT_DIR}/.." && pwd)/models/GO-1"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"
echo -e "\033[33m[INFO] CUDA_HOME: ${CUDA_HOME}\033[0m"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"

repo_id="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
lerobot_output_dir="${SCRIPT_DIR}/data"
default_lerobot_data_path="${lerobot_output_dir}/${repo_id}"
lerobot_data_path="${LEROBOT_DATA_PATH:-${default_lerobot_data_path}}"
cfg_path="${GO1_CFG_PATH:-go1/configs/go1_sft_xpolicylab.py}"

if [ -n "${LEROBOT_DATA_PATH:-}" ]; then
    echo -e "\033[33m[INFO] Using external LeRobot dataset path: ${lerobot_data_path}\033[0m"
    if [ ! -d "${lerobot_data_path}" ]; then
        echo -e "\033[31m[ERROR] External LeRobot dataset path does not exist: ${lerobot_data_path}\033[0m"
        exit 1
    fi
else
    echo -e "\033[33m[INFO] Checking if LeRobot dataset exists at: ${lerobot_data_path}\033[0m"
    if [ -d "${lerobot_data_path}" ]; then
        echo -e "\033[33m[INFO] LeRobot dataset '${repo_id}' already exists, skipping conversion.\033[0m"
    else
        echo -e "\033[33m[INFO] Converting HDF5 data to LeRobot format...\033[0m"
        bash "${SCRIPT_DIR}/process_data.sh" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" 30 "${lerobot_output_dir}"
    fi
fi

RUN_BASENAME="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
RUNNAME="${RUNNAME:-${RUN_BASENAME}-${RUN_TIMESTAMP}}"
export RUN_BASENAME RUNNAME
export DATA_ROOT_DIR="${lerobot_data_path}"
export ACTION_DIM="${action_dim}"
export STATE_DIM="${action_dim}"
export CTRL_FREQ="${CTRL_FREQ:-30}"
export ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-30}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_GO1_MODEL_PATH}}"
export DEFAULT_PROMPT="Do your job."
export TRAIN_SEED="${seed}"

IFS=',' read -ra GPU_ARRAY <<< "${gpu_id}"
NPROC=${#GPU_ARRAY[@]}
export NPROC_PER_NODE="${NPROC}"
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo -e "\033[33m[INFO] Starting GO1 training with RUNNAME=${RUNNAME}, NPROC=${NPROC}, MASTER_PORT=${MASTER_PORT}\033[0m"
echo -e "\033[33m[INFO] Training config path: ${cfg_path}\033[0m"
echo -e "\033[33m[INFO] DATA_ROOT_DIR: ${DATA_ROOT_DIR}\033[0m"
echo -e "\033[33m[INFO] CTRL_FREQ: ${CTRL_FREQ}, ACTION_CHUNK_SIZE: ${ACTION_CHUNK_SIZE}\033[0m"

export PYTHONPATH="${AGIBOT_DIR}:${PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-go1}"
export WANDB_NAME="${RUNNAME}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/tmp}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${TMPDIR}"

cd "${AGIBOT_DIR}"
CKPT_DIR="${SCRIPT_DIR}/checkpoints/${RUNNAME}"
if [ "${OVERWRITE_RUN_DIR:-False}" = "True" ] && [ -d "${CKPT_DIR}" ]; then
    echo -e "\033[33m[INFO] OVERWRITE_RUN_DIR=True, removing old checkpoint dir: ${CKPT_DIR}\033[0m"
    rm -rf "${CKPT_DIR}"
fi
mkdir -p "${CKPT_DIR}/log"
echo "${CKPT_DIR}" > "${SCRIPT_DIR}/checkpoints/${RUN_BASENAME}.latest"

torchrun \
    --nnodes=1 \
    --node-rank=0 \
    --master-addr=127.0.0.1 \
    --nproc-per-node="${NPROC}" \
    --master-port="${MASTER_PORT}" \
    go1/internvl/train/go1_train.py \
    --cfg_path "${cfg_path}" \
    2>&1 | tee -a "${CKPT_DIR}/log/training_$(date +"%Y%m%d_%H%M").txt"

echo -e "\033[33m[INFO] Training complete. Checkpoints saved to: ${CKPT_DIR}\033[0m"
