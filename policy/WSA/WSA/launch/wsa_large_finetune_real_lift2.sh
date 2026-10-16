#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6392}
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
export LEROBOT_DDP_FIND_UNUSED_PARAMETERS=false

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
DEFAULT_POLICY_INIT_PATH="${DEFAULT_POLICY_INIT_PATH:-zaleni/WSA-Large}"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-${PRETRAINED_MODEL_DIR:-${DEFAULT_POLICY_INIT_PATH}}}}"
SKIP_DIT_LOAD_FROM_PRETRAIN="${SKIP_DIT_LOAD_FROM_PRETRAIN:-true}"
NATIVE_WSA_LARGE_CHECKPOINT_PATH="${NATIVE_WSA_LARGE_CHECKPOINT_PATH:-}"
LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER:-false}"

DATASET_DIR="${DATASET_DIR:-/path/to/plasticbottle_v30}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_DIR}")}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
USE_DIST_LOADING="${USE_DIST_LOADING:-false}"
USE_RAMDISK_DATASET="${USE_RAMDISK_DATASET:-true}"
RAMDISK_ROOT="${RAMDISK_ROOT:-/dev/shm/${USER:-$(id -un)}/wsa_large_datasets}"
RAMDISK_REQUIRED_PERCENT="${RAMDISK_REQUIRED_PERCENT:-120}"
CACHE_IN_MEMORY="${CACHE_IN_MEMORY:-false}"

ACTION_TYPE="${ACTION_TYPE:-abs}"
ACTION_DIM="${ACTION_DIM:-14}"
PROPRIO_DIM="${PROPRIO_DIM:-14}"
RAW_ACTION_DIM="${RAW_ACTION_DIM:-14}"
RAW_PROPRIO_DIM="${RAW_PROPRIO_DIM:-14}"
NUM_FRAMES="${NUM_FRAMES:-49}"
ACTION_HORIZON="${ACTION_HORIZON:-$((NUM_FRAMES - 1))}"
N_ACTION_STEPS="${N_ACTION_STEPS:-$((NUM_FRAMES - 1))}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-6}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-384}"
VIDEO_WIDTH="${VIDEO_WIDTH:-320}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-robotwin}"
STANDARDIZE_VIDEO_SIZE_BY_CAMERAS="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS:-true}"
NORM_DEFAULT_MODE="${NORM_DEFAULT_MODE:-q01q99}"
ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"
IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"

if [[ -z "${USE_EXTERNAL_STATS+x}" ]]; then
  if [[ "${ACTION_TYPE}" == "delta" ]]; then
    USE_EXTERNAL_STATS=true
  else
    USE_EXTERNAL_STATS=false
  fi
fi
NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-}"

BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-42000}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-${STEPS}}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-2}"
if [[ -z "${NUM_WORKERS+x}" ]]; then
  if [[ "${CACHE_IN_MEMORY}" == "true" ]]; then
    NUM_WORKERS=0
  else
    NUM_WORKERS=12
  fi
fi

LR="${LR:-7.5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DECAY_LR="${DECAY_LR:-2.0e-5}"
LAMBDA_VIDEO="${LAMBDA_VIDEO:-0.1}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
MASK_ACTION_DIM_PADDING_LOSS="${MASK_ACTION_DIM_PADDING_LOSS:-true}"
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
WSA_LARGE_CHECKPOINT_MIXED_ATTN="${WSA_LARGE_CHECKPOINT_MIXED_ATTN:-true}"

IMAGE_KEYS="${IMAGE_KEYS:-[\"head\",\"left\",\"right\"]}"
IMAGE_RAW_SHAPES="${IMAGE_RAW_SHAPES:-[[3,480,640],[3,480,640],[3,480,640]]}"
IMAGE_SHAPES="${IMAGE_SHAPES:-[[3,240,320],[3,240,320],[3,240,320]]}"
ACTION_KEYS="${ACTION_KEYS:-[\"default\"]}"
ACTION_RAW_SHAPES="${ACTION_RAW_SHAPES:-[${RAW_ACTION_DIM}]}"
ACTION_SHAPES="${ACTION_SHAPES:-[${RAW_ACTION_DIM}]}"
STATE_KEYS="${STATE_KEYS:-[\"default\"]}"
STATE_RAW_SHAPES="${STATE_RAW_SHAPES:-[${RAW_PROPRIO_DIM}]}"
STATE_SHAPES="${STATE_SHAPES:-[${RAW_PROPRIO_DIM}]}"
PROCESSOR_DELTA_ACTION_DIM_MASK="${PROCESSOR_DELTA_ACTION_DIM_MASK:-{\"default\":[true,true,true,true,true,true,false,true,true,true,true,true,true,false]}}"

case "${ACTION_TYPE}" in
  abs|delta)
    ;;
  *)
    echo "Unsupported ACTION_TYPE=${ACTION_TYPE}. Expected abs or delta."
    exit 1
    ;;
esac

case "${USE_EXTERNAL_STATS}" in
  true|false)
    ;;
  *)
    echo "Unsupported USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}. Expected true or false."
    exit 1
    ;;
esac

case "${CACHE_IN_MEMORY}" in
  true|false)
    ;;
  *)
    echo "Unsupported CACHE_IN_MEMORY=${CACHE_IN_MEMORY}. Expected true or false."
    exit 1
    ;;
esac

case "${USE_RAMDISK_DATASET}" in
  true|false)
    ;;
  *)
    echo "Unsupported USE_RAMDISK_DATASET=${USE_RAMDISK_DATASET}. Expected true or false."
    exit 1
    ;;
esac

case "${SKIP_DIT_LOAD_FROM_PRETRAIN}" in
  true|false)
    ;;
  *)
    echo "Unsupported SKIP_DIT_LOAD_FROM_PRETRAIN=${SKIP_DIT_LOAD_FROM_PRETRAIN}. Expected true or false."
    exit 1
    ;;
esac

if [[ "${USE_EXTERNAL_STATS}" == "true" && -z "${NORMALIZATION_STATS_PATH}" ]]; then
  if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    NORMALIZATION_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH}"
  else
    NORMALIZATION_STATS_PATH="${NORM_STATS_ROOT}/${ACTION_TYPE}/${DATASET_NAME}/stats.json"
  fi
fi

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

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR does not exist: ${DATASET_DIR}"
  exit 1
fi

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "meta/info.json not found under DATASET_DIR: ${DATASET_DIR}"
  exit 1
fi

SOURCE_DATASET_DIR="${DATASET_DIR}"
SOURCE_DATASET_REPO_ID="${DATASET_REPO_ID}"
if [[ "${USE_RAMDISK_DATASET}" == "true" ]]; then
  RAMDISK_DATASET_DIR="${RAMDISK_DATASET_DIR:-${RAMDISK_ROOT}/${DATASET_NAME}}"
  if [[ "${DATASET_DIR}" != "${RAMDISK_DATASET_DIR}" ]]; then
    mkdir -p "${RAMDISK_ROOT}"
    if [[ ! -w "${RAMDISK_ROOT}" ]]; then
      echo "RAMDISK_ROOT is not writable: ${RAMDISK_ROOT}"
      echo "If running in Docker, use --shm-size or set RAMDISK_ROOT to a writable tmpfs mount."
      exit 1
    fi
    source_kb="$(du -sk "${SOURCE_DATASET_DIR}" | awk '{print $1}')"
    existing_kb=0
    if [[ -d "${RAMDISK_DATASET_DIR}" ]]; then
      existing_kb="$(du -sk "${RAMDISK_DATASET_DIR}" | awk '{print $1}')"
    fi
    available_kb="$(df -Pk "${RAMDISK_ROOT}" | awk 'NR==2 {print $4}')"
    effective_available_kb=$((available_kb + existing_kb))
    required_kb=$(((source_kb * RAMDISK_REQUIRED_PERCENT + 99) / 100))
    echo "RAM disk check: root=${RAMDISK_ROOT}, dataset=${source_kb} KiB, available=${available_kb} KiB, existing_target=${existing_kb} KiB, required=${required_kb} KiB"
    if (( effective_available_kb < required_kb )); then
      echo "Not enough RAM disk space for dataset copy."
      echo "In Docker, start with e.g. --shm-size=64g or --ipc=host, or set USE_RAMDISK_DATASET=false/RAMDISK_ROOT=<large tmpfs>."
      exit 1
    fi
    echo "Copying dataset to RAM disk: ${SOURCE_DATASET_DIR} -> ${RAMDISK_DATASET_DIR}"
    mkdir -p "${RAMDISK_DATASET_DIR}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "${SOURCE_DATASET_DIR}/" "${RAMDISK_DATASET_DIR}/"
    else
      cp -a "${SOURCE_DATASET_DIR}/." "${RAMDISK_DATASET_DIR}/"
    fi
    DATASET_DIR="${RAMDISK_DATASET_DIR}"
  fi
fi
if [[ -z "${SOURCE_DATASET_REPO_ID}" || "${SOURCE_DATASET_REPO_ID}" == "${SOURCE_DATASET_DIR}" ]]; then
  DATASET_REPO_ID="${DATASET_DIR}"
fi

robot_type="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["robot_type"])' \
    "${DATASET_DIR}/meta/info.json"
)"

if [[ "${robot_type}" != "real_lift2" ]]; then
  echo "Expected robot_type=real_lift2, got ${robot_type}"
  echo "Please reconvert the dataset or update meta/info.json."
  exit 1
fi

if [[ "${SKIP_DIT_LOAD_FROM_PRETRAIN}" != "true" && ! -f "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
  echo "Missing ActionDiT backbone: ${ACTION_DIT_PRETRAINED_PATH}"
  echo "Generate WSA_Large expert backbones first, or set ACTION_DIT_PRETRAINED_PATH."
  exit 1
fi

if [[ "${SKIP_DIT_LOAD_FROM_PRETRAIN}" != "true" && ! -f "${FUTURE_3D_PRETRAINED_PATH}" ]]; then
  echo "Missing Future3DExpert backbone: ${FUTURE_3D_PRETRAINED_PATH}"
  echo "Generate WSA_Large expert backbones first, or set FUTURE_3D_PRETRAINED_PATH."
  exit 1
fi

if [[ -n "${POLICY_INIT_PATH}" ]]; then
  if [[ -e "${POLICY_INIT_PATH}" && ! -d "${POLICY_INIT_PATH}" ]]; then
    echo "POLICY_INIT_PATH exists but is not a directory: ${POLICY_INIT_PATH}"
    exit 1
  fi
  if [[ -d "${POLICY_INIT_PATH}" ]]; then
    if [[ ! -f "${POLICY_INIT_PATH}/config.json" ]]; then
      echo "Missing policy config: ${POLICY_INIT_PATH}/config.json"
      exit 1
    fi
    if [[ ! -f "${POLICY_INIT_PATH}/model.safetensors" ]]; then
      echo "Missing policy weights: ${POLICY_INIT_PATH}/model.safetensors"
      exit 1
    fi
  fi
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" && ! -f "${NORMALIZATION_STATS_PATH}" ]]; then
  echo "Missing normalization stats: ${NORMALIZATION_STATS_PATH}"
  echo "Compute them first with:"
  echo "  python tools/compute_norm_stats_single.py --repo_id \"${DATASET_DIR}\" --action_mode \"${ACTION_TYPE}\" --chunk_size \"${ACTION_HORIZON}\" --output_dir \"${NORM_STATS_ROOT}\""
  exit 1
fi

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating Real_Lift2 dataset mappings..."
  python -c 'import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
image_keys = json.loads(sys.argv[2])
features = set(info.get("features", {}).keys())
required = {"observation.state", "action"}
required.update(f"observation.images.{key}" for key in image_keys)
missing = sorted(required - features)
print("robot_type={}, features={}".format(info.get("robot_type"), len(features)))
if missing:
    raise SystemExit(f"Missing required WSA_Large Real_Lift2 features: {missing}")
' "${DATASET_DIR}/meta/info.json" "${IMAGE_KEYS}"
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs_real/${POLICY}}"
BOOTSTRAP_TAG="${BOOTSTRAP_TAG:-pretrained300k}"
JOB_NAME="${JOB_NAME:-${WSA_LARGE_VARIANT}-real_lift2-${DATASET_NAME}-${ACTION_TYPE}-${BOOTSTRAP_TAG}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_ID}" > "${REPO_ID_FILE}"

if [[ "${LOAD_TEXT_ENCODER}" == "true" ]]; then
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
else
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJ_ROOT}/outputs/WSA_Large/text_embeds/real_lift2/${DATASET_NAME}}"
fi
TEXT_EMBED_CACHE_MAX_ENTRIES="${TEXT_EMBED_CACHE_MAX_ENTRIES:-0}"

if [[ "${LOAD_TEXT_ENCODER}" != "true" && ! -d "${TEXT_EMBED_CACHE_DIR}" ]]; then
  echo "LOAD_TEXT_ENCODER=false but TEXT_EMBED_CACHE_DIR does not exist: ${TEXT_EMBED_CACHE_DIR}"
  echo "Precompute text embeddings with:"
  echo "  python tools/precompute_text_embeds.py --repo-id-file \"${REPO_ID_FILE}\" --text-embedding-cache-dir \"${TEXT_EMBED_CACHE_DIR}\" --device cuda"
  echo "Or set LOAD_TEXT_ENCODER=true."
  exit 1
fi

echo "WSA_LARGE_VARIANT=${WSA_LARGE_VARIANT}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "SOURCE_DATASET_DIR=${SOURCE_DATASET_DIR}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "USE_RAMDISK_DATASET=${USE_RAMDISK_DATASET}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "robot_type=${robot_type}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "ACTION_DIM=${ACTION_DIM}, PROPRIO_DIM=${PROPRIO_DIM}"
echo "RAW_ACTION_DIM=${RAW_ACTION_DIM}, RAW_PROPRIO_DIM=${RAW_PROPRIO_DIM}"
echo "ACTION_HORIZON=${ACTION_HORIZON}, N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "NORMALIZATION_STATS_PATH=${NORMALIZATION_STATS_PATH:-<metadata-or-auto>}"
echo "TEXT_EMBED_CACHE_DIR=${TEXT_EMBED_CACHE_DIR:-<text-encoder-runtime>}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH:-<none>}"
echo "SKIP_DIT_LOAD_FROM_PRETRAIN=${SKIP_DIT_LOAD_FROM_PRETRAIN}"
echo "CACHE_IN_MEMORY=${CACHE_IN_MEMORY}, USE_DIST_LOADING=${USE_DIST_LOADING}, NUM_WORKERS=${NUM_WORKERS}"
echo "EVAL_FREQ=${EVAL_FREQ}, EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES}"
echo "VIDEO_SIZE=[${VIDEO_HEIGHT},${VIDEO_WIDTH}], CONCAT_MULTI_CAMERA=${CONCAT_MULTI_CAMERA}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}, IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "MASK_ACTION_DIM_PADDING_LOSS=${MASK_ACTION_DIM_PADDING_LOSS}"
echo "Future3D: LAMBDA_3D=${LAMBDA_3D}, DA3_NUM_VIEWS=${DA3_NUM_VIEWS}, TOKENS_PER_VIEW=${FUTURE_3D_TOKENS_PER_VIEW}, VIEW_LAYOUT=${FUTURE_3D_VIEW_ATTENTION_LAYOUT}"
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
    --policy.skip_dit_load_from_pretrain="${SKIP_DIT_LOAD_FROM_PRETRAIN}"
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
    --policy.mask_action_dim_padding_loss="${MASK_ACTION_DIM_PADDING_LOSS}"
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
    --dataset.repo_id="real_lift2_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats="${USE_EXTERNAL_STATS}"
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
    --dataset.context_len=128
    --dataset.val_set_proportion=0.0
    --dataset.skip_padding_as_possible=false
    --dataset.concat_multi_camera="${CONCAT_MULTI_CAMERA}"
    --dataset.processor_norm_default_mode="${NORM_DEFAULT_MODE}"
    --dataset.processor_num_output_cameras="${PROCESSOR_NUM_OUTPUT_CAMERAS}"
    --dataset.processor_action_output_dim="${ACTION_DIM}"
    --dataset.processor_proprio_output_dim="${PROPRIO_DIM}"
    --dataset.processor_delta_action_dim_mask="${PROCESSOR_DELTA_ACTION_DIM_MASK}"
    --dataset.future_3d_target_index="${FUTURE_3D_TARGET_INDEX}"
    --dataset.cache_in_memory="${CACHE_IN_MEMORY}"

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --eval_freq="${EVAL_FREQ}"
    --eval_max_batches="${EVAL_MAX_BATCHES}"

    --wandb.enable=true
    --wandb.project=WSA_Large_RealLift2
    --wandb.mode=${WANDB_MODE}
)

if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
    ARGS+=(--dataset.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}")
fi
ARGS+=(--dataset.text_embedding_cache_max_entries="${TEXT_EMBED_CACHE_MAX_ENTRIES}")

if [[ -n "${POLICY_INIT_PATH}" ]]; then
    ARGS+=(--policy.pretrained_path="${POLICY_INIT_PATH}")
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

if [[ -n "${NORMALIZATION_STATS_PATH}" ]]; then
    ARGS+=(--dataset.normalization_stats_path="${NORMALIZATION_STATS_PATH}")
fi

if [[ -n "${NATIVE_WSA_LARGE_CHECKPOINT_PATH}" ]]; then
    ARGS+=(--policy.native_checkpoint_path="${NATIVE_WSA_LARGE_CHECKPOINT_PATH}")
fi

if [[ -n "${VIDEO_BACKEND}" ]]; then
    ARGS+=(--dataset.video_backend="${VIDEO_BACKEND}")
fi

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ "${USE_DIST_LOADING}" == "true" ]]; then
    ARGS+=(--dataset.dist_loading=true)
fi

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"
    )
fi

accelerate launch "${ARGS[@]}"
