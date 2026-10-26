#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

# export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# WANDB_TOKEN=${WANDB_TOKEN}
# CONDA_ROOT=${_CONDA_ROOT}
# CONDA_ENV=qwenaction

# source ${CONDA_ROOT}/etc/profile.d/conda.sh
# conda activate ${CONDA_ENV}

# wandb login ${WANDB_TOKEN}

###############################################################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

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
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

POLICY="qwenaction"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-}"
WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-}"
USE_DIST_LOADING="${USE_DIST_LOADING:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${GRAD_ACCUM_STEPS:-1}}"
STEPS="${STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-25}"
NUM_WORKERS="${NUM_WORKERS:-12}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"

if [[ -n "${POLICY_INIT_PATH}" && ! -d "${POLICY_INIT_PATH}" ]]; then
  echo "POLICY_INIT_PATH does not exist or is not a directory: ${POLICY_INIT_PATH}"
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

if [[ -n "${WEIGHT_RULES_PATH}" && ! -f "${WEIGHT_RULES_PATH}" ]]; then
  echo "WEIGHT_RULES_PATH does not exist: ${WEIGHT_RULES_PATH}"
  exit 1
fi

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  find -L "${root}" -path "*/meta/info.json" 2>/dev/null \
    | while read -r info_path; do
        ds_dir="$(dirname "$(dirname "${info_path}")")"
        if [[ -d "${ds_dir}/data" || -d "${ds_dir}/videos" ]]; then
          echo "${ds_dir}"
        fi
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${ROBOTWIN_ROOT}")

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No valid RoboTwin LeRobot datasets found under ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
  exit 1
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" && -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
  echo "USE_EXTERNAL_STATS=true but neither DATASET_EXTERNAL_STATS_PATH nor DATASET_EXTERNAL_STATS_ROOT is set."
  exit 1
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
INIT_TAG="scratch"
if [[ -n "${POLICY_INIT_PATH}" ]]; then
  INIT_TAG="${BOOTSTRAP_TAG:-internvla_a1_3b}"
fi
JOB_NAME="${JOB_NAME:-${POLICY}-robotwin-${ACTION_TYPE}-chunk${CHUNK_SIZE}-${INIT_TAG}-action_only-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "Discovered ${#DATASET_REPO_IDS[@]} RoboTwin datasets under ${ROBOTWIN_ROOT}"
echo "INIT_TAG=${INIT_TAG}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH:-<scratch>}"
echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
echo "DATASET_EXTERNAL_STATS_ROOT=${DATASET_EXTERNAL_STATS_ROOT}"
echo "WEIGHT_RULES_PATH=${WEIGHT_RULES_PATH:-<none>}"
echo "USE_DIST_LOADING=${USE_DIST_LOADING}"
echo "BATCH_SIZE(per_device)=${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "STEPS=${STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "LOG_FREQ=${LOG_FREQ}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "JOB_NAME=${JOB_NAME}"
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

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"
    --policy.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5.0e-5
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5.0e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=QwenAction
    --wandb.mode="${WANDB_MODE}"
)

if [[ -n "${POLICY_INIT_PATH}" ]]; then
    ARGS+=(--policy.pretrained_path="${POLICY_INIT_PATH}")
fi

if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    ARGS+=(--dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}")
fi

if [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    ARGS+=(--dataset.external_stats_root="${DATASET_EXTERNAL_STATS_ROOT}")
fi

if [[ -n "${WEIGHT_RULES_PATH}" ]]; then
    ARGS+=(--dataset.weight_rules_path="${WEIGHT_RULES_PATH}")
fi

if [[ -n "${VIDEO_BACKEND}" ]]; then
    ARGS+=(--dataset.video_backend="${VIDEO_BACKEND}")
fi

if [[ "${USE_DIST_LOADING}" == "true" ]]; then
    ARGS+=(--dataset.dist_loading=true)
fi

accelerate launch "${ARGS[@]}"
