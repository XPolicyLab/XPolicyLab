#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

# export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# WANDB_TOKEN=${WANDB_TOKEN}
# CONDA_ROOT=${_CONDA_ROOT}
# CONDA_ENV=wsa

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

# export CUDA_HOME="/usr/local/cuda-12.8"
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

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

POLICY="WSA_Base"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-zaleni/WSA-Base}}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/DA3-LARGE-1-1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
ENABLE_3D_QUERIES="${ENABLE_3D_QUERIES:-false}"
NUM_3D_QUERY_TOKENS="${NUM_3D_QUERY_TOKENS:-432}"
WSA_BASE_ATTENTION_MASK_MODE="${WSA_BASE_ATTENTION_MASK_MODE:-default}"
LAMBDA_GEN="${LAMBDA_GEN:-${GEN_LAMBDA:-0.01}}"
LAMBDA_3D="${LAMBDA_3D:-0.01}"

USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-}"
WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-}"
USE_DIST_LOADING="${USE_DIST_LOADING:-false}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${GRAD_ACCUM_STEPS:-1}}"
STEPS="${STEPS:-200000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-50}"
NUM_WORKERS="${NUM_WORKERS:-12}"

if [[ "${WSA_BASE_ATTENTION_MASK_MODE}" == "causal" ]]; then
  echo "WSA_BASE_ATTENTION_MASK_MODE=causal detected; using causal."
  WSA_BASE_ATTENTION_MASK_MODE="causal"
fi

if [[ -n "${POLICY_INIT_PATH}" && -e "${POLICY_INIT_PATH}" && ! -d "${POLICY_INIT_PATH}" ]]; then
  echo "POLICY_INIT_PATH exists but is not a directory: ${POLICY_INIT_PATH}"
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

if [[ "${WSA_BASE_ATTENTION_MASK_MODE}" == "causal" && "${ENABLE_3D_QUERIES}" != "true" ]]; then
  echo "WSA_BASE_ATTENTION_MASK_MODE=causal requires ENABLE_3D_QUERIES=true."
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

case "${USE_EXTERNAL_STATS}" in
  true|false)
    ;;
  *)
    echo "USE_EXTERNAL_STATS must be true or false, got ${USE_EXTERNAL_STATS}"
    exit 1
    ;;
esac

# A single aggregated JSON applies to every discovered repo and belongs in
# DATASET_EXTERNAL_STATS_PATH. Keep this compatibility conversion because it
# is easy to accidentally pass that file through the older *_ROOT variable.
if [[ -z "${DATASET_EXTERNAL_STATS_PATH}" && -n "${DATASET_EXTERNAL_STATS_ROOT}" && -f "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
  echo "DATASET_EXTERNAL_STATS_ROOT points to a file; treating it as DATASET_EXTERNAL_STATS_PATH."
  DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_ROOT}"
  DATASET_EXTERNAL_STATS_ROOT=""
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
  if [[ -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    echo "USE_EXTERNAL_STATS=true but neither DATASET_EXTERNAL_STATS_PATH nor DATASET_EXTERNAL_STATS_ROOT is set."
    exit 1
  fi
  if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" && ! -f "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    echo "DATASET_EXTERNAL_STATS_PATH does not exist: ${DATASET_EXTERNAL_STATS_PATH}"
    exit 1
  fi
  if [[ -z "${DATASET_EXTERNAL_STATS_PATH}" && -n "${DATASET_EXTERNAL_STATS_ROOT}" && ! -d "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    echo "DATASET_EXTERNAL_STATS_ROOT does not exist or is not a directory: ${DATASET_EXTERNAL_STATS_ROOT}"
    exit 1
  fi
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
INIT_TAG="scratch"
if [[ -n "${POLICY_INIT_PATH}" ]]; then
  INIT_TAG="${BOOTSTRAP_TAG:-pretrained}"
fi
JOB_NAME="${JOB_NAME:-${POLICY}-robotwin-${ACTION_TYPE}-chunk${CHUNK_SIZE}-${INIT_TAG}-${WSA_BASE_ATTENTION_MASK_MODE}-gen${LAMBDA_GEN}-3d${LAMBDA_3D}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${JOB_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-WSA_Base}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "Discovered ${#DATASET_REPO_IDS[@]} RoboTwin datasets under ${ROBOTWIN_ROOT}"
echo "INIT_TAG=${INIT_TAG}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH:-<scratch>}"
echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
echo "COSMOS_TOKENIZER_PATH_OR_NAME=${COSMOS_TOKENIZER_PATH_OR_NAME}"
echo "DA3_MODEL_PATH_OR_NAME=${DA3_MODEL_PATH_OR_NAME}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "ENABLE_3D_QUERIES=${ENABLE_3D_QUERIES}"
echo "NUM_3D_QUERY_TOKENS=${NUM_3D_QUERY_TOKENS}"
echo "WSA_BASE_ATTENTION_MASK_MODE=${WSA_BASE_ATTENTION_MASK_MODE}"
echo "LAMBDA_GEN=${LAMBDA_GEN}"
echo "LAMBDA_3D=${LAMBDA_3D}"
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
echo "JOB_NAME=${JOB_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"

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
    --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5e-6
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
    --policy.lambda_gen="${LAMBDA_GEN}"
    --policy.lambda_3d="${LAMBDA_3D}"
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"
    --policy.log_da3_teacher_timing=true

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
    --wandb.project="${WANDB_PROJECT}"
    --wandb.mode="${WANDB_MODE}"
)

if [[ -n "${POLICY_INIT_PATH}" ]]; then
    ARGS+=(--policy.pretrained_path="${POLICY_INIT_PATH}")
fi

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
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
