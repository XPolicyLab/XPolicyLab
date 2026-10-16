#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

WANDB_TOKEN="${WANDB_TOKEN:-}"
CONDA_ROOT="${_CONDA_ROOT:-${CONDA_ROOT:-}}"
CONDA_ENV="${CONDA_ENV:-mot}"

if [[ -n "${CONDA_ROOT}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

if [[ -n "${WANDB_TOKEN}" ]]; then
    wandb login "${WANDB_TOKEN}"
fi

###############################################################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_HOME
export LD_LIBRARY_PATH="${CUDA_HOME:+${CUDA_HOME}/lib64:}${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"
cd "${PROJ_ROOT}"

if (( $# > 3 )); then
  echo "Usage:"
  echo "  bash launch/wsa_base_finetune.sh DATASET_REPO_ID [ACTION_TYPE] [USE_EXTERNAL_STATS]"
  echo "  DATASET_REPO_ID=/path/or/hf_repo POLICY_INIT_PATH=/path/to/bootstrap bash launch/wsa_base_finetune.sh"
  exit 1
fi

POLICY="${POLICY:-WSA_Base}"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-}}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-nvidia/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-depth-anything/DA3-LARGE-1.1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"

DATASET_REPO_ID="${1:-${DATASET_REPO_ID:-}}"
ACTION_TYPE="${2:-${ACTION_TYPE:-delta}}"
POSITIONAL_USE_EXTERNAL_STATS="${3:-}"

if [[ -z "${DATASET_REPO_ID}" ]]; then
  echo "Please provide DATASET_REPO_ID as the first argument or environment variable."
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

if [[ -n "${POSITIONAL_USE_EXTERNAL_STATS}" ]]; then
  USE_EXTERNAL_STATS="${POSITIONAL_USE_EXTERNAL_STATS}"
elif [[ -z "${USE_EXTERNAL_STATS+x}" ]]; then
  if [[ "${ACTION_TYPE}" == "delta" ]]; then
    USE_EXTERNAL_STATS=true
  else
    USE_EXTERNAL_STATS=false
  fi
fi

case "${USE_EXTERNAL_STATS}" in
  true|false)
    ;;
  *)
    echo "USE_EXTERNAL_STATS must be true or false, got ${USE_EXTERNAL_STATS}"
    exit 1
    ;;
esac

if [[ -z "${POLICY_INIT_PATH}" ]]; then
  echo "Please set POLICY_INIT_PATH to the WSA Base bootstrap checkpoint."
  echo "For backward compatibility, PRETRAINED_PATH is also accepted."
  exit 1
fi

if [[ -z "${DATASET_NAME:-}" ]]; then
  if [[ -e "${DATASET_REPO_ID}" ]]; then
    DATASET_NAME="$(basename "${DATASET_REPO_ID}")"
  else
    DATASET_NAME="${DATASET_REPO_ID//[\/ ]/_}"
  fi
fi

CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
WSA_BASE_ATTENTION_MASK_MODE="${WSA_BASE_ATTENTION_MASK_MODE:-causal}"
ENABLE_3D_QUERIES="${ENABLE_3D_QUERIES:-true}"
NUM_3D_QUERY_TOKENS="${NUM_3D_QUERY_TOKENS:-432}"
LAMBDA_3D="${LAMBDA_3D:-0.01}"

NORM_STATS_ROOT="${NORM_STATS_ROOT:-norm_stats}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
if [[ "${USE_EXTERNAL_STATS}" == "true" && -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
  DATASET_EXTERNAL_STATS_PATH="${NORM_STATS_ROOT}/${ACTION_TYPE}/${DATASET_NAME}/stats.json"
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
  if [[ -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    echo "USE_EXTERNAL_STATS=true but neither DATASET_EXTERNAL_STATS_PATH nor DATASET_EXTERNAL_STATS_ROOT is set."
    exit 1
  fi
  if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" && ! -f "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    echo "Missing external stats: ${DATASET_EXTERNAL_STATS_PATH}"
    echo "Compute them first with tools/compute_norm_stats_single.py or set DATASET_EXTERNAL_STATS_PATH."
    exit 1
  fi
fi

ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"
IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"

BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-120000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-50}"
NUM_WORKERS="${NUM_WORKERS:-12}"

OPTIMIZER_LR="${OPTIMIZER_LR:-5.0e-5}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-5.0e-6}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
JOB_NAME="${JOB_NAME:-${POLICY}-${DATASET_NAME}-${ACTION_TYPE}-chunk${CHUNK_SIZE}-attn-${WSA_BASE_ATTENTION_MASK_MODE}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${JOB_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot_lab_${POLICY}}"

echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "WSA_BASE_ATTENTION_MASK_MODE=${WSA_BASE_ATTENTION_MASK_MODE}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH:-<unset>}"
echo "DATASET_EXTERNAL_STATS_ROOT=${DATASET_EXTERNAL_STATS_ROOT:-<unset>}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}"
echo "IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "STEPS=${STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers="${NUM_WORKERS}"
    --job_name="${JOB_NAME}"

    --policy.type="${POLICY}"
    --policy.repo_id="lerobot_lab/${POLICY}"
    --policy.pretrained_path="${POLICY_INIT_PATH}"
    --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"
    --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr="${OPTIMIZER_LR}"
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
    --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.attention_mask_mode="${WSA_BASE_ATTENTION_MASK_MODE}"
    --policy.enable_3d_queries="${ENABLE_3D_QUERIES}"
    --policy.num_3d_query_tokens="${NUM_3D_QUERY_TOKENS}"
    --policy.lambda_3d="${LAMBDA_3D}"
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"
    --policy.log_da3_teacher_timing=true

    --dataset.type="${POLICY}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats="${USE_EXTERNAL_STATS}"

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT}"
    --wandb.mode="${WANDB_MODE}"
)

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" && -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    ARGS+=(--dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}")
fi

if [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    ARGS+=(--dataset.external_stats_root="${DATASET_EXTERNAL_STATS_ROOT}")
fi

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"
    )
fi

accelerate launch "${ARGS[@]}"
