#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

WANDB_TOKEN="${WANDB_TOKEN:-}"
CONDA_ROOT="${_CONDA_ROOT:-${CONDA_ROOT:-}}"
CONDA_ENV="${CONDA_ENV:-internvla_a1}"

if [[ -n "${CONDA_ROOT}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

if [[ -n "${WANDB_TOKEN}" ]]; then
    wandb login "${WANDB_TOKEN}"
fi

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 1. policy config
POLICY="pi05"
PRETRAINED_PATH="lerobot/pi05_base"

# 2. dataset config
DATASET_REPO_ID="$1"
ACTION_TYPE=${2:-abs}          # abs | delta
USE_EXTERNAL_STATS=${3:-false} # true | false

# 3. output config
BASE_OUTPUT_DIR="outputs/${POLICY}"
PRETRAINED_DETAIL="pi05_base"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-${DATASET_REPO_ID//[\/ ]/_}-${ACTION_TYPE}-${PRETRAINED_DETAIL}-finetune"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}" 
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers=16
    --job_name="${JOB_NAME}"

    # ---- Policy ----
    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path=${PRETRAINED_PATH}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=2.5e-5
    --policy.scheduler_warmup_steps=1000
    --policy.scheduler_decay_steps=30000
    --policy.scheduler_decay_lr=2.5e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false

    # ---- Dataset ----
    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=true
    --dataset.external_stats_path=${HF_HOME}/lerobot/stats/${ACTION_TYPE}/${DATASET_REPO_ID}/stats.json

    # ---- Training ----
    --seed=42
    --batch_size=16
    --steps=30000
    --save_freq=30000
    --log_freq=200

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=lerobot_lab_${POLICY}
    --wandb.mode=offline
)

accelerate launch "${ARGS[@]}"
