#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6391}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJ_ROOT}"

POLICY="${POLICY:-WSA_Large}"
WSA_LARGE_VARIANT="${WSA_LARGE_VARIANT:-wsa_large}"
case "${WSA_LARGE_VARIANT}" in
  wsa_large|wsa_large_joint)
    ;;
  *)
    echo "Unsupported WSA_Large variant=${WSA_LARGE_VARIANT}. Expected wsa_large or wsa_large_joint."
    exit 1
    ;;
esac

WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
WSA_LARGE_REDIRECT_COMMON_FILES="${WSA_LARGE_REDIRECT_COMMON_FILES:-true}"
WSA_LARGE_ASSET_ROOT="${WSA_LARGE_ASSET_ROOT:-${PROJ_ROOT}/checkpoints/wsa_large}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${WSA_LARGE_ASSET_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
FUTURE_3D_PRETRAINED_PATH="${FUTURE_3D_PRETRAINED_PATH:-${WSA_LARGE_ASSET_ROOT}/Future3DExpert_linear_interp_Wan22_alphascale_768hdim.pt}"
NATIVE_WSA_LARGE_CHECKPOINT_PATH="${NATIVE_WSA_LARGE_CHECKPOINT_PATH:-}"
LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER:-false}"

INTERNDATA_ROOT="${INTERNDATA_ROOT:-/path/to/InternData-A1-v30}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/RoboTwin-LeRobot-v30}"
ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-/path/to/Robochallengev3.0_eef}"
AGIBOT_ROOT="${AGIBOT_ROOT:-/path/to/Agibotv3.0}"
EGODEX_LEROBOT_ROOT="${EGODEX_LEROBOT_ROOT:-/path/to/Egodex_v_taskrepos_v30}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-norm_stats_32}"
WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-configs/weight_rules_wsa_large_pretrain.yaml}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-false}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
USE_DIST_LOADING="${USE_DIST_LOADING:-true}"
PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD:-true}"
REPO_ASSIGNMENT_LOG_MODE="${REPO_ASSIGNMENT_LOG_MODE:-summary}"
REPO_ASSIGNMENT_LOG_LIMIT="${REPO_ASSIGNMENT_LOG_LIMIT:-4}"
WEIGHT_RULES_DEFAULT_GROUP_MODE="${WEIGHT_RULES_DEFAULT_GROUP_MODE:-error}"
WEIGHT_RULES_DEFAULT_GROUP_LIMIT="${WEIGHT_RULES_DEFAULT_GROUP_LIMIT:-20}"
ADAPTER_LOG_MODE="${ADAPTER_LOG_MODE:-summary}"
ADAPTER_LOG_LIMIT="${ADAPTER_LOG_LIMIT:-3}"
RANK_DEVICE_LOG="${RANK_DEVICE_LOG:-true}"
DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-3600}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

export LEROBOT_DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC}"
export LEROBOT_DDP_FIND_UNUSED_PARAMETERS="${LEROBOT_DDP_FIND_UNUSED_PARAMETERS:-${DDP_FIND_UNUSED_PARAMETERS}}"
export LEROBOT_PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD}"
export LEROBOT_REPO_ASSIGNMENT_LOG_MODE="${REPO_ASSIGNMENT_LOG_MODE}"
export LEROBOT_REPO_ASSIGNMENT_LOG_LIMIT="${REPO_ASSIGNMENT_LOG_LIMIT}"
export LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_MODE="${WEIGHT_RULES_DEFAULT_GROUP_MODE}"
export LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_LIMIT="${WEIGHT_RULES_DEFAULT_GROUP_LIMIT}"
export LEROBOT_WSA_LARGE_ADAPTER_LOG_MODE="${ADAPTER_LOG_MODE}"
export LEROBOT_WSA_LARGE_ADAPTER_LOG_LIMIT="${ADAPTER_LOG_LIMIT}"
export LEROBOT_LOG_RANK_DEVICE_MAP="${RANK_DEVICE_LOG}"

if [[ "${LOAD_TEXT_ENCODER}" == "true" ]]; then
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
else
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJ_ROOT}/outputs/WSA_Large/text_embeds/pretrain}"
fi
TEXT_EMBED_CACHE_MAX_ENTRIES="${TEXT_EMBED_CACHE_MAX_ENTRIES:-0}"
TEXT_EMBED_CONTEXT_LEN="${TEXT_EMBED_CONTEXT_LEN:-128}"

ACTION_TYPE="${ACTION_TYPE:-delta}"
ACTION_DIM="${ACTION_DIM:-24}"
PROPRIO_DIM="${PROPRIO_DIM:-24}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-384}"
VIDEO_WIDTH="${VIDEO_WIDTH:-320}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-robotwin}"
STANDARDIZE_VIDEO_SIZE_BY_CAMERAS="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS:-true}"
NORM_DEFAULT_MODE="${NORM_DEFAULT_MODE:-z-score}"
ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"
IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"

BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-300000}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-300000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"

LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
DECAY_LR="${DECAY_LR:-1.0e-5}"
LAMBDA_VIDEO="${LAMBDA_VIDEO:-1.0}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
LAMBDA_3D="${LAMBDA_3D:-0.1}"
DA3_NUM_VIEWS="${DA3_NUM_VIEWS:-3}"
PROCESSOR_NUM_OUTPUT_CAMERAS="${PROCESSOR_NUM_OUTPUT_CAMERAS:-${DA3_NUM_VIEWS}}"
FUTURE_3D_TOKENS_PER_VIEW="${FUTURE_3D_TOKENS_PER_VIEW:-144}"
FUTURE_3D_VIEW_ATTENTION_LAYOUT="${FUTURE_3D_VIEW_ATTENTION_LAYOUT:-${CONCAT_MULTI_CAMERA}}"
FUTURE_3D_QUERY_MODE="${FUTURE_3D_QUERY_MODE:-slot_noise}"
FUTURE_3D_QUERY_NOISE_SCALE="${FUTURE_3D_QUERY_NOISE_SCALE:-0.5}"
FUTURE_3D_QUERY_NOISE_MIN_SIGMA="${FUTURE_3D_QUERY_NOISE_MIN_SIGMA:-0.0}"
FUTURE_3D_QUERY_NOISE_MAX_SIGMA="${FUTURE_3D_QUERY_NOISE_MAX_SIGMA:-0.5}"
FUTURE_3D_QUERY_SIGMA_SOURCE="${FUTURE_3D_QUERY_SIGMA_SOURCE:-constant}"
FUTURE_3D_SLOT_POS_SCALE="${FUTURE_3D_SLOT_POS_SCALE:-0.5}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-depth-anything/DA3-LARGE-1.1}"
DA3_VARIANT="${DA3_VARIANT:-large}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
DA3_TEACHER_PROCESS_RES="${DA3_TEACHER_PROCESS_RES:-504}"
LOG_DA3_TEACHER_TIMING="${LOG_DA3_TEACHER_TIMING:-true}"
FUTURE_3D_TARGET_INDEX="${FUTURE_3D_TARGET_INDEX:--1}"
DTYPE="${DTYPE:-bfloat16}"
WSA_LARGE_CHECKPOINT_MIXED_ATTN="${WSA_LARGE_CHECKPOINT_MIXED_ATTN:-false}"

IMAGE_KEYS="${IMAGE_KEYS:-[\"image0\",\"image1\",\"image2\"]}"
IMAGE_RAW_SHAPES="${IMAGE_RAW_SHAPES:-[[3,224,224],[3,224,224],[3,224,224]]}"
IMAGE_SHAPES="${IMAGE_SHAPES:-[[3,224,224],[3,224,224],[3,224,224]]}"
ACTION_KEYS="${ACTION_KEYS:-[\"default\"]}"
ACTION_RAW_SHAPES="${ACTION_RAW_SHAPES:-[${ACTION_DIM}]}"
ACTION_SHAPES="${ACTION_SHAPES:-[${ACTION_DIM}]}"
STATE_KEYS="${STATE_KEYS:-[\"default\"]}"
STATE_RAW_SHAPES="${STATE_RAW_SHAPES:-[${PROPRIO_DIM}]}"
STATE_SHAPES="${STATE_SHAPES:-[${PROPRIO_DIM}]}"

case "${ACTION_TYPE}" in
  abs|delta)
    ;;
  *)
    echo "Unsupported ACTION_TYPE=${ACTION_TYPE}. Expected abs or delta."
    exit 1
    ;;
esac

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

mapfile -t DATASET_REPO_IDS < <(
  {
    discover_dataset_dirs "${INTERNDATA_ROOT}"
    discover_dataset_dirs "${ROBOTWIN_ROOT}"
    discover_dataset_dirs "${ROBOCHALLENGE_ROOT}"
    discover_dataset_dirs "${AGIBOT_ROOT}"
    discover_dataset_dirs "${EGODEX_LEROBOT_ROOT}"
  } | sort -u
)

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No valid LeRobot datasets found."
  echo "Please set one or more of: INTERNDATA_ROOT ROBOTWIN_ROOT ROBOCHALLENGE_ROOT AGIBOT_ROOT EGODEX_LEROBOT_ROOT"
  exit 1
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
JOB_NAME="${POLICY}-${WSA_LARGE_VARIANT}-multidata-${ACTION_TYPE}-pretrain-$(date +'%Y_%m_%d_%H_%M_%S')"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

if [[ ! -f "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
  echo "Missing ActionDiT backbone: ${ACTION_DIT_PRETRAINED_PATH}"
  exit 1
fi

if [[ ! -f "${FUTURE_3D_PRETRAINED_PATH}" ]]; then
  echo "Missing Future3DExpert backbone: ${FUTURE_3D_PRETRAINED_PATH}"
  exit 1
fi

if [[ -z "${DATASET_EXTERNAL_STATS_ROOT}" || ! -d "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
  echo "DATASET_EXTERNAL_STATS_ROOT does not exist: ${DATASET_EXTERNAL_STATS_ROOT}"
  exit 1
fi

if [[ -n "${WEIGHT_RULES_PATH}" && ! -f "${WEIGHT_RULES_PATH}" ]]; then
  echo "WEIGHT_RULES_PATH does not exist: ${WEIGHT_RULES_PATH}"
  exit 1
fi

if [[ "${LOAD_TEXT_ENCODER}" != "true" && ! -d "${TEXT_EMBED_CACHE_DIR}" ]]; then
  echo "LOAD_TEXT_ENCODER=false but TEXT_EMBED_CACHE_DIR does not exist: ${TEXT_EMBED_CACHE_DIR}"
  echo "Precompute text embeddings with:"
  echo "  python tools/precompute_text_embeds.py --repo-id-file \"${REPO_ID_FILE}\" --text-embedding-cache-dir \"${TEXT_EMBED_CACHE_DIR}\" --context-len \"${TEXT_EMBED_CONTEXT_LEN}\" --device cuda"
  echo "Or set LOAD_TEXT_ENCODER=true."
  exit 1
fi

if [[ "${LOAD_TEXT_ENCODER}" != "true" ]]; then
  echo "Verifying text embedding cache coverage..."
  python tools/precompute_text_embeds.py \
    --repo-id-file "${REPO_ID_FILE}" \
    --text-embedding-cache-dir "${TEXT_EMBED_CACHE_DIR}" \
    --context-len "${TEXT_EMBED_CONTEXT_LEN}" \
    --verify-cache-only true
fi

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating dataset robot_type registration and stats readiness..."
  for ds_dir in "${DATASET_REPO_IDS[@]}"; do
    info_path="${ds_dir}/meta/info.json"
    python -c 'from lerobot.transforms.constants import get_feature_mapping, get_image_mapping, get_mask_mapping, infer_embodiment_variant; import json, pathlib, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
features = info.get("features", {})
robot_type = info["robot_type"]
resolved = infer_embodiment_variant(robot_type, features)
get_feature_mapping(robot_type, features)
get_image_mapping(robot_type, features)
get_mask_mapping(robot_type, features)
print(f"{sys.argv[2]} -> robot_type={robot_type}, resolved={resolved}, features={len(features)}")
' "${info_path}" "${ds_dir}"
  done
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

echo "Discovered ${#DATASET_REPO_IDS[@]} datasets"
echo "INTERNDATA_ROOT=${INTERNDATA_ROOT}"
echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
echo "ROBOCHALLENGE_ROOT=${ROBOCHALLENGE_ROOT}"
echo "AGIBOT_ROOT=${AGIBOT_ROOT}"
echo "EGODEX_LEROBOT_ROOT=${EGODEX_LEROBOT_ROOT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "ACTION_DIM=${ACTION_DIM}, PROPRIO_DIM=${PROPRIO_DIM}"
echo "DATASET_EXTERNAL_STATS_ROOT=${DATASET_EXTERNAL_STATS_ROOT}"
echo "WEIGHT_RULES_PATH=${WEIGHT_RULES_PATH:-<disabled>}"
echo "TEXT_EMBED_CACHE_DIR=${TEXT_EMBED_CACHE_DIR:-<text-encoder-on-the-fly>}"
echo "TEXT_EMBED_CONTEXT_LEN=${TEXT_EMBED_CONTEXT_LEN}"
echo "PARALLEL_DATASET_LOAD=${PARALLEL_DATASET_LOAD}"
echo "REPO_ASSIGNMENT_LOG_MODE=${REPO_ASSIGNMENT_LOG_MODE}, REPO_ASSIGNMENT_LOG_LIMIT=${REPO_ASSIGNMENT_LOG_LIMIT}"
echo "WEIGHT_RULES_DEFAULT_GROUP_MODE=${WEIGHT_RULES_DEFAULT_GROUP_MODE}, WEIGHT_RULES_DEFAULT_GROUP_LIMIT=${WEIGHT_RULES_DEFAULT_GROUP_LIMIT}"
echo "ADAPTER_LOG_MODE=${ADAPTER_LOG_MODE}, ADAPTER_LOG_LIMIT=${ADAPTER_LOG_LIMIT}"
echo "RANK_DEVICE_LOG=${RANK_DEVICE_LOG}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}, IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH}"
echo "FUTURE_3D_PRETRAINED_PATH=${FUTURE_3D_PRETRAINED_PATH}"
echo "NUM_FRAMES=${NUM_FRAMES}, ACTION_HORIZON=${ACTION_HORIZON}, ACTION_VIDEO_FREQ_RATIO=${ACTION_VIDEO_FREQ_RATIO}"
echo "VIDEO_SIZE=[${VIDEO_HEIGHT},${VIDEO_WIDTH}], CONCAT_MULTI_CAMERA=${CONCAT_MULTI_CAMERA}"
echo "NUM_EPOCHS=${NUM_EPOCHS:-<disabled>}"
echo "TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-<disabled>}"
echo "USE_DIST_LOADING=${USE_DIST_LOADING}, DDP_TIMEOUT_SEC=${DDP_TIMEOUT_SEC}"
echo "DDP_FIND_UNUSED_PARAMETERS=${LEROBOT_DDP_FIND_UNUSED_PARAMETERS}"
echo "Future3D: LAMBDA_3D=${LAMBDA_3D}, DA3_NUM_VIEWS=${DA3_NUM_VIEWS}, TOKENS_PER_VIEW=${FUTURE_3D_TOKENS_PER_VIEW}, VIEW_LAYOUT=${FUTURE_3D_VIEW_ATTENTION_LAYOUT}"
echo "Future3D query: MODE=${FUTURE_3D_QUERY_MODE}, NOISE_SCALE=${FUTURE_3D_QUERY_NOISE_SCALE}, SIGMA=[${FUTURE_3D_QUERY_NOISE_MIN_SIGMA},${FUTURE_3D_QUERY_NOISE_MAX_SIGMA}], SIGMA_SOURCE=${FUTURE_3D_QUERY_SIGMA_SOURCE}, SLOT_POS_SCALE=${FUTURE_3D_SLOT_POS_SCALE}"
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
    --policy.variant="${WSA_LARGE_VARIANT}"
    --policy.model_id="${WAN_MODEL_ID}"
    --policy.tokenizer_model_id="${WAN_TOKENIZER_MODEL_ID}"
    --policy.redirect_common_files="${WSA_LARGE_REDIRECT_COMMON_FILES}"
    --policy.action_dit_pretrained_path="${ACTION_DIT_PRETRAINED_PATH}"
    --policy.future_3d_pretrained_path="${FUTURE_3D_PRETRAINED_PATH}"
    --policy.load_text_encoder="${LOAD_TEXT_ENCODER}"
    --policy.dtype="${DTYPE}"
    --policy.mot_checkpoint_mixed_attn="${WSA_LARGE_CHECKPOINT_MIXED_ATTN}"
    --policy.action_dim="${ACTION_DIM}"
    --policy.proprio_dim="${PROPRIO_DIM}"
    --policy.action_horizon="${ACTION_HORIZON}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.num_inference_steps="${NUM_INFERENCE_STEPS}"
    --policy.lambda_video="${LAMBDA_VIDEO}"
    --policy.lambda_action="${LAMBDA_ACTION}"
    --policy.lambda_3d="${LAMBDA_3D}"
    --policy.da3_num_views="${DA3_NUM_VIEWS}"
    --policy.future_3d_tokens_per_view="${FUTURE_3D_TOKENS_PER_VIEW}"
    --policy.future_3d_view_attention_layout="${FUTURE_3D_VIEW_ATTENTION_LAYOUT}"
    --policy.future_3d_query_mode="${FUTURE_3D_QUERY_MODE}"
    --policy.future_3d_query_noise_scale="${FUTURE_3D_QUERY_NOISE_SCALE}"
    --policy.future_3d_query_noise_min_sigma="${FUTURE_3D_QUERY_NOISE_MIN_SIGMA}"
    --policy.future_3d_query_noise_max_sigma="${FUTURE_3D_QUERY_NOISE_MAX_SIGMA}"
    --policy.future_3d_query_sigma_source="${FUTURE_3D_QUERY_SIGMA_SOURCE}"
    --policy.future_3d_slot_pos_scale="${FUTURE_3D_SLOT_POS_SCALE}"
    --policy.future_3d_target_index="${FUTURE_3D_TARGET_INDEX}"
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_teacher_process_res="${DA3_TEACHER_PROCESS_RES}"
    --policy.log_da3_teacher_timing="${LOG_DA3_TEACHER_TIMING}"
    --policy.action_norm_default_mode="${NORM_DEFAULT_MODE}"
    --policy.optimizer_lr="${LR}"
    --policy.optimizer_weight_decay="${WEIGHT_DECAY}"

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.pretrain_multi_embodiment=true
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=true
    --dataset.external_stats_root="${DATASET_EXTERNAL_STATS_ROOT}"
    --dataset.image_keys="${IMAGE_KEYS}"
    --dataset.image_raw_shapes="${IMAGE_RAW_SHAPES}"
    --dataset.image_shapes="${IMAGE_SHAPES}"
    --dataset.action_keys="${ACTION_KEYS}"
    --dataset.action_raw_shapes="${ACTION_RAW_SHAPES}"
    --dataset.action_shapes="${ACTION_SHAPES}"
    --dataset.state_keys="${STATE_KEYS}"
    --dataset.state_raw_shapes="${STATE_RAW_SHAPES}"
    --dataset.state_shapes="${STATE_SHAPES}"
    --dataset.num_frames="${NUM_FRAMES}"
    --dataset.action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO}"
    --dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]"
    --dataset.standardize_video_size_by_cameras="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
    --dataset.context_len="${TEXT_EMBED_CONTEXT_LEN}"
    --dataset.val_set_proportion=0.0
    --dataset.skip_padding_as_possible=false
    --dataset.concat_multi_camera="${CONCAT_MULTI_CAMERA}"
    --dataset.processor_norm_default_mode="${NORM_DEFAULT_MODE}"
    --dataset.processor_num_output_cameras="${PROCESSOR_NUM_OUTPUT_CAMERAS}"
    --dataset.processor_action_output_dim="${ACTION_DIM}"
    --dataset.processor_proprio_output_dim="${PROPRIO_DIM}"
    --dataset.future_3d_target_index="${FUTURE_3D_TARGET_INDEX}"
    --dataset.video_backend="${VIDEO_BACKEND}"

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=WSA_Large
    --wandb.mode=${WANDB_MODE}
)

if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
    ARGS+=(--dataset.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}")
fi
ARGS+=(--dataset.text_embedding_cache_max_entries="${TEXT_EMBED_CACHE_MAX_ENTRIES}")

if [[ -n "${WEIGHT_RULES_PATH}" ]]; then
    ARGS+=(--dataset.weight_rules_path="${WEIGHT_RULES_PATH}")
fi

if [[ "${USE_DIST_LOADING}" == "true" ]]; then
    ARGS+=(--dataset.dist_loading=true)
fi

if [[ -n "${WARMUP_STEPS}" ]]; then
    ARGS+=(--policy.scheduler_warmup_steps="${WARMUP_STEPS}")
fi

if [[ -n "${DECAY_LR}" ]]; then
    ARGS+=(--policy.scheduler_decay_lr="${DECAY_LR}")
fi

if [[ -n "${NUM_EPOCHS}" ]]; then
    ARGS+=(--policy.train_num_epochs="${NUM_EPOCHS}")
fi

if [[ -n "${TRAIN_MAX_STEPS}" ]]; then
    ARGS+=(--policy.train_max_steps="${TRAIN_MAX_STEPS}")
fi

if [[ -n "${NATIVE_WSA_LARGE_CHECKPOINT_PATH}" ]]; then
    ARGS+=(--policy.native_checkpoint_path="${NATIVE_WSA_LARGE_CHECKPOINT_PATH}")
fi

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"
    )
fi

accelerate launch "${ARGS[@]}"
