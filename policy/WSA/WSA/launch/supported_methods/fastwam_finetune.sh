#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6389}"
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
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJ_ROOT}"

POLICY="fastwam"
FASTWAM_VARIANT="${FASTWAM_VARIANT:-fastwam}"

WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
FASTWAM_REDIRECT_COMMON_FILES="${FASTWAM_REDIRECT_COMMON_FILES:-true}"
FASTWAM_ASSET_ROOT="${FASTWAM_ASSET_ROOT:-${PROJ_ROOT}/checkpoints/fastwam}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${FASTWAM_ASSET_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
NATIVE_FASTWAM_CHECKPOINT_PATH="${NATIVE_FASTWAM_CHECKPOINT_PATH:-}"
LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER:-false}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [[ "${LOAD_TEXT_ENCODER}" == "true" ]]; then
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
else
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJ_ROOT}/outputs/fastwam/text_embeds/robotwin}"
fi
NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH:-}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
USE_DIST_LOADING="${USE_DIST_LOADING:-false}"

NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-384}"
VIDEO_WIDTH="${VIDEO_WIDTH:-320}"
CONTEXT_LEN="${CONTEXT_LEN:-128}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-25}"
NUM_WORKERS="${NUM_WORKERS:-8}"

LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
DECAY_LR="${DECAY_LR:-}"
LAMBDA_VIDEO="${LAMBDA_VIDEO:-1.0}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
DTYPE="${DTYPE:-bfloat16}"
FASTWAM_CHECKPOINT_MIXED_ATTN="${FASTWAM_CHECKPOINT_MIXED_ATTN:-false}"

ACTION_DIM="${ACTION_DIM:-14}"
PROPRIO_DIM="${PROPRIO_DIM:-14}"
PROCESSOR_NUM_OUTPUT_CAMERAS="${PROCESSOR_NUM_OUTPUT_CAMERAS:-3}"
PROCESSOR_DELTA_ACTION_DIM_MASK="${PROCESSOR_DELTA_ACTION_DIM_MASK:-{\"default\":[true,true,true,true,true,true,false,true,true,true,true,true,true,false]}}"

case "${DTYPE}" in
  bfloat16)
    ACCELERATE_MIXED_PRECISION="bf16"
    ;;
  float16)
    ACCELERATE_MIXED_PRECISION="fp16"
    ;;
  float32)
    ACCELERATE_MIXED_PRECISION="no"
    ;;
  *)
    echo "Unsupported DTYPE=${DTYPE}. Expected one of: bfloat16, float16, float32"
    exit 1
    ;;
esac

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

if [[ ! -f "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
  echo "Missing ActionDiT backbone: ${ACTION_DIT_PRETRAINED_PATH}"
  echo "Generate it with:"
  echo "  python src/lerobot/scripts/fastwam_preprocess_action_dit_backbone.py --output \"${ACTION_DIT_PRETRAINED_PATH}\" --device cuda --dtype bfloat16"
  exit 1
fi

if [[ "${LOAD_TEXT_ENCODER}" != "true" && ! -d "${TEXT_EMBED_CACHE_DIR}" ]]; then
  echo "LOAD_TEXT_ENCODER=false but TEXT_EMBED_CACHE_DIR does not exist: ${TEXT_EMBED_CACHE_DIR}"
  echo "Precompute text embeddings with:"
  echo "  python src/lerobot/scripts/fastwam_precompute_text_embeds.py --repo-id-file <repo_id_file.txt> --text-embedding-cache-dir \"${TEXT_EMBED_CACHE_DIR}\" --device cuda"
  echo "Or set LOAD_TEXT_ENCODER=true."
  exit 1
fi

if [[ -n "${NORMALIZATION_STATS_PATH}" && ! -f "${NORMALIZATION_STATS_PATH}" ]]; then
  echo "NORMALIZATION_STATS_PATH does not exist: ${NORMALIZATION_STATS_PATH}"
  exit 1
fi

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating RoboTwin dataset mappings..."
  for ds_dir in "${DATASET_REPO_IDS[@]}"; do
    info_path="${ds_dir}/meta/info.json"
    python -c 'import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
features = info.get("features", {})
keys = set(features)
required = {
    "observation.state",
    "action",
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}
missing = sorted(required - keys)
codebase_version = info.get("codebase_version", "unknown")
robot_type = info.get("robot_type", "unknown")
print(f"{sys.argv[2]} -> codebase={codebase_version}, robot_type={robot_type}, features={len(keys)}")
if codebase_version != "v3.0":
    raise SystemExit(f"Dataset is not v3.0: {sys.argv[2]}")
if missing:
    raise SystemExit(f"Dataset is missing RoboTwin features {missing}: {sys.argv[2]}")
' "${info_path}" "${ds_dir}"
  done
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

echo "Discovered ${#DATASET_REPO_IDS[@]} RoboTwin datasets under ${ROBOTWIN_ROOT}"
printf '  %s\n' "${DATASET_REPO_IDS[@]}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
JOB_NAME="${JOB_NAME:-${POLICY}-${FASTWAM_VARIANT}-robotwin-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "FASTWAM_VARIANT=${FASTWAM_VARIANT}"
echo "WAN_MODEL_ID=${WAN_MODEL_ID}"
echo "WAN_TOKENIZER_MODEL_ID=${WAN_TOKENIZER_MODEL_ID}"
echo "ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH}"
echo "NATIVE_FASTWAM_CHECKPOINT_PATH=${NATIVE_FASTWAM_CHECKPOINT_PATH:-<none>}"
echo "LOAD_TEXT_ENCODER=${LOAD_TEXT_ENCODER}"
echo "TEXT_EMBED_CACHE_DIR=${TEXT_EMBED_CACHE_DIR:-<none>}"
echo "NORMALIZATION_STATS_PATH=${NORMALIZATION_STATS_PATH:-<none>}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "ACTION_HORIZON=${ACTION_HORIZON}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "ACTION_VIDEO_FREQ_RATIO=${ACTION_VIDEO_FREQ_RATIO}"
echo "VIDEO_SIZE=[${VIDEO_HEIGHT},${VIDEO_WIDTH}]"
echo "ACTION_DIM=${ACTION_DIM}"
echo "PROPRIO_DIM=${PROPRIO_DIM}"
echo "PROCESSOR_NUM_OUTPUT_CAMERAS=${PROCESSOR_NUM_OUTPUT_CAMERAS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

ARGS=(
    --multi_gpu
    --mixed_precision="${ACCELERATE_MIXED_PRECISION}"
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
    --policy.push_to_hub=false
    --policy.variant="${FASTWAM_VARIANT}"
    --policy.model_id="${WAN_MODEL_ID}"
    --policy.tokenizer_model_id="${WAN_TOKENIZER_MODEL_ID}"
    --policy.redirect_common_files="${FASTWAM_REDIRECT_COMMON_FILES}"
    --policy.action_dit_pretrained_path="${ACTION_DIT_PRETRAINED_PATH}"
    --policy.load_text_encoder="${LOAD_TEXT_ENCODER}"
    --policy.dtype="${DTYPE}"
    --policy.mot_checkpoint_mixed_attn="${FASTWAM_CHECKPOINT_MIXED_ATTN}"
    --policy.action_dim="${ACTION_DIM}"
    --policy.proprio_dim="${PROPRIO_DIM}"
    --policy.action_horizon="${ACTION_HORIZON}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.num_inference_steps="${NUM_INFERENCE_STEPS}"
    --policy.lambda_video="${LAMBDA_VIDEO}"
    --policy.lambda_action="${LAMBDA_ACTION}"
    --policy.optimizer_lr="${LR}"
    --policy.optimizer_weight_decay="${WEIGHT_DECAY}"
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
    --policy.train_num_epochs="${NUM_EPOCHS}"

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.image_keys="[cam_high,cam_left_wrist,cam_right_wrist]"
    --dataset.image_raw_shapes="[[3,224,224],[3,224,224],[3,224,224]]"
    --dataset.image_shapes="[[3,224,224],[3,224,224],[3,224,224]]"
    --dataset.action_raw_shapes="[${ACTION_DIM}]"
    --dataset.action_shapes="[${ACTION_DIM}]"
    --dataset.state_raw_shapes="[${PROPRIO_DIM}]"
    --dataset.state_shapes="[${PROPRIO_DIM}]"
    --dataset.num_frames="${NUM_FRAMES}"
    --dataset.action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO}"
    --dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]"
    --dataset.context_len="${CONTEXT_LEN}"
    --dataset.val_set_proportion=0.0
    --dataset.skip_padding_as_possible=false
    --dataset.concat_multi_camera=robotwin
    --dataset.processor_num_output_cameras="${PROCESSOR_NUM_OUTPUT_CAMERAS}"
    --dataset.processor_action_output_dim="${ACTION_DIM}"
    --dataset.processor_proprio_output_dim="${PROPRIO_DIM}"
    --dataset.processor_delta_action_dim_mask="${PROCESSOR_DELTA_ACTION_DIM_MASK}"

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=FastWAM
    --wandb.mode="${WANDB_MODE}"
)

if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
    ARGS+=(--dataset.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}")
fi

if [[ -n "${DECAY_LR}" ]]; then
    ARGS+=(--policy.scheduler_decay_lr="${DECAY_LR}")
fi

if [[ -n "${TRAIN_MAX_STEPS}" ]]; then
    ARGS+=(--policy.train_max_steps="${TRAIN_MAX_STEPS}")
fi

if [[ -n "${NORMALIZATION_STATS_PATH}" ]]; then
    ARGS+=(--dataset.normalization_stats_path="${NORMALIZATION_STATS_PATH}")
fi

if [[ -n "${NATIVE_FASTWAM_CHECKPOINT_PATH}" ]]; then
    ARGS+=(--policy.native_checkpoint_path="${NATIVE_FASTWAM_CHECKPOINT_PATH}")
fi

if [[ -n "${VIDEO_BACKEND}" ]]; then
    ARGS+=(--dataset.video_backend="${VIDEO_BACKEND}")
fi

if [[ "${USE_DIST_LOADING}" == "true" ]]; then
    echo "USE_DIST_LOADING=true is not supported for FastWAM in this framework."
    echo "Leave USE_DIST_LOADING=false so Accelerate can shard the dataloader correctly."
    exit 1
fi

accelerate launch "${ARGS[@]}"
