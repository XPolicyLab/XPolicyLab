#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-7390}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))
TRAIN_SEED="${TRAIN_SEED:-42}"

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
DEFAULT_POLICY_INIT_PATH="${DEFAULT_POLICY_INIT_PATH:-zaleni/WSA-Large}"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-${PRETRAINED_MODEL_DIR:-${DEFAULT_POLICY_INIT_PATH}}}}"
SKIP_DIT_LOAD_FROM_PRETRAIN="${SKIP_DIT_LOAD_FROM_PRETRAIN:-false}"
NATIVE_WSA_LARGE_CHECKPOINT_PATH="${NATIVE_WSA_LARGE_CHECKPOINT_PATH:-}"
LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER:-false}"
RESUME="${RESUME:-false}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"
RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/RoboTwin-LeRobot-v30}"
ROBOTWIN_REQUIRE_THREE_CAMERAS="${ROBOTWIN_REQUIRE_THREE_CAMERAS:-true}"
if [[ "${LOAD_TEXT_ENCODER}" == "true" ]]; then
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
else
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJ_ROOT}/outputs/WSA_Large/text_embeds/robotwin}"
fi
TEXT_EMBED_CACHE_MAX_ENTRIES="${TEXT_EMBED_CACHE_MAX_ENTRIES:-0}"
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-norm_stats_32}"
DATASET_EXTERNAL_STATS_ROBOT_TYPE="${DATASET_EXTERNAL_STATS_ROBOT_TYPE:-aloha}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
USE_DIST_LOADING="${USE_DIST_LOADING:-false}"

ACTION_TYPE="${ACTION_TYPE:-abs}"
ACTION_DIM="${ACTION_DIM:-14}"
PROPRIO_DIM="${PROPRIO_DIM:-14}"
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
STEPS="${STEPS:-}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
NUM_WORKERS="${NUM_WORKERS:-16}"

LR="${LR:-6.0e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DECAY_LR="${DECAY_LR:-1.0e-6}"
LAMBDA_VIDEO="${LAMBDA_VIDEO:-1.0}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
MASK_ACTION_DIM_PADDING_LOSS="${MASK_ACTION_DIM_PADDING_LOSS:-true}"
LAMBDA_3D="${LAMBDA_3D:-0.1}"
if [[ -z "${LEROBOT_DDP_FIND_UNUSED_PARAMETERS:-}" ]]; then
  if python -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)' "${LAMBDA_3D}"; then
    export LEROBOT_DDP_FIND_UNUSED_PARAMETERS=false
  else
    export LEROBOT_DDP_FIND_UNUSED_PARAMETERS=true
  fi
else
  export LEROBOT_DDP_FIND_UNUSED_PARAMETERS
fi
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

IMAGE_KEYS="${IMAGE_KEYS:-[\"cam_high\",\"cam_left_wrist\",\"cam_right_wrist\"]}"
IMAGE_RAW_SHAPES="${IMAGE_RAW_SHAPES:-[[3,480,640],[3,480,640],[3,480,640]]}"
IMAGE_SHAPES="${IMAGE_SHAPES:-[[3,240,320],[3,240,320],[3,240,320]]}"
ACTION_KEYS="${ACTION_KEYS:-[\"default\"]}"
ACTION_RAW_SHAPES="${ACTION_RAW_SHAPES:-[14]}"
ACTION_SHAPES="${ACTION_SHAPES:-[14]}"
STATE_KEYS="${STATE_KEYS:-[\"default\"]}"
STATE_RAW_SHAPES="${STATE_RAW_SHAPES:-[14]}"
STATE_SHAPES="${STATE_SHAPES:-[14]}"
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

case "${SKIP_DIT_LOAD_FROM_PRETRAIN}" in
  true|false)
    ;;
  *)
    echo "Unsupported SKIP_DIT_LOAD_FROM_PRETRAIN=${SKIP_DIT_LOAD_FROM_PRETRAIN}. Expected true or false."
    exit 1
    ;;
esac

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
  if [[ -z "${NORMALIZATION_STATS_PATH}" ]]; then
    if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
      NORMALIZATION_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH}"
    elif [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
      NORMALIZATION_STATS_PATH="${DATASET_EXTERNAL_STATS_ROOT}/${DATASET_EXTERNAL_STATS_ROBOT_TYPE}/${ACTION_TYPE}/stats.json"
    fi
  fi

  if [[ -z "${NORMALIZATION_STATS_PATH}" ]]; then
    echo "USE_EXTERNAL_STATS=true but no normalization stats path could be resolved."
    echo "Set NORMALIZATION_STATS_PATH, DATASET_EXTERNAL_STATS_PATH, or DATASET_EXTERNAL_STATS_ROOT."
    exit 1
  fi
else
  NORMALIZATION_STATS_PATH=""
fi

ACTION_STATS_PATH="${ACTION_STATS_PATH:-${POLICY_ACTION_STATS_PATH:-}}"
if [[ -z "${ACTION_STATS_PATH}" && -n "${NORMALIZATION_STATS_PATH}" ]]; then
  ACTION_STATS_PATH="${NORMALIZATION_STATS_PATH}"
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

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  find -L "${root}" -path "*/meta/info.json" 2>/dev/null \
    | while read -r info_path; do
        ds_dir="$(dirname "$(dirname "${info_path}")")"
        if [[ ! -d "${ds_dir}/data" && ! -d "${ds_dir}/videos" ]]; then
          continue
        fi
        if [[ "${ROBOTWIN_REQUIRE_THREE_CAMERAS}" == "true" ]]; then
          python -c 'import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
image_keys = json.loads(sys.argv[2])
features = set(info.get("features", {}).keys())
required = {"observation.state", "action"}
required.update(f"observation.images.{key}" for key in image_keys)
raise SystemExit(0 if required.issubset(features) else 1)
' "${info_path}" "${IMAGE_KEYS}" || continue
        fi
        echo "${ds_dir}"
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${ROBOTWIN_ROOT}")

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No valid RoboTwin LeRobot datasets found under ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
  if [[ "${ROBOTWIN_REQUIRE_THREE_CAMERAS}" == "true" ]]; then
    echo "The default filter keeps only datasets with IMAGE_KEYS=${IMAGE_KEYS}."
    echo "Set ROBOTWIN_REQUIRE_THREE_CAMERAS=false if you intentionally want to include other layouts."
  fi
  exit 1
fi

if [[ "${SKIP_DIT_LOAD_FROM_PRETRAIN}" != "true" && ! -f "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
    echo "Missing ActionDiT backbone: ${ACTION_DIT_PRETRAINED_PATH}"
    echo "Generate WSA_Large expert backbones with:"
    echo "  python tools/preprocess_expert_backbones.py --expert both --action-output \"${ACTION_DIT_PRETRAINED_PATH}\" --future-3d-output \"${FUTURE_3D_PRETRAINED_PATH}\" --action-dim ${ACTION_DIM} --da3-num-views ${DA3_NUM_VIEWS} --future-3d-tokens-per-view ${FUTURE_3D_TOKENS_PER_VIEW} --device cuda --dtype bfloat16"
    exit 1
fi

if [[ "${SKIP_DIT_LOAD_FROM_PRETRAIN}" != "true" && ! -f "${FUTURE_3D_PRETRAINED_PATH}" ]]; then
    echo "Missing Future3DExpert backbone: ${FUTURE_3D_PRETRAINED_PATH}"
    echo "Generate WSA_Large expert backbones with:"
    echo "  python tools/preprocess_expert_backbones.py --expert both --action-output \"${ACTION_DIT_PRETRAINED_PATH}\" --future-3d-output \"${FUTURE_3D_PRETRAINED_PATH}\" --action-dim ${ACTION_DIM} --da3-num-views ${DA3_NUM_VIEWS} --future-3d-tokens-per-view ${FUTURE_3D_TOKENS_PER_VIEW} --device cuda --dtype bfloat16"
    exit 1
fi

if [[ -n "${POLICY_INIT_PATH}" ]]; then
  if [[ "${RESUME}" == "true" && -n "${RESUME_CHECKPOINT_DIR}" ]]; then
    :
  else
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
fi

RESUME_CONFIG_JOB_NAME="${RESUME_CONFIG_JOB_NAME:-}"
RESUME_CONFIG_OUTPUT_DIR="${RESUME_CONFIG_OUTPUT_DIR:-}"
if [[ "${RESUME}" == "true" ]]; then
  if [[ -z "${RESUME_CHECKPOINT_DIR}" ]]; then
    if [[ -z "${POLICY_INIT_PATH}" ]]; then
      echo "RESUME=true requires RESUME_CHECKPOINT_DIR or POLICY_INIT_PATH."
      exit 1
    fi
    if [[ "$(basename "${POLICY_INIT_PATH%/}")" != "pretrained_model" ]]; then
      echo "RESUME=true expects POLICY_INIT_PATH to point to a pretrained_model directory:"
      echo "  ${POLICY_INIT_PATH}"
      exit 1
    fi
    RESUME_CHECKPOINT_DIR="$(dirname "${POLICY_INIT_PATH%/}")"
  fi

  if [[ -z "${RESUME_CONFIG_PATH}" ]]; then
    RESUME_CONFIG_PATH="${RESUME_CHECKPOINT_DIR%/}/pretrained_model/train_config.json"
  fi

  if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
    echo "Resume config not found: ${RESUME_CONFIG_PATH}"
    exit 1
  fi

  if [[ ! -d "${RESUME_CHECKPOINT_DIR%/}/training_state" ]]; then
    echo "Missing training_state under checkpoint: ${RESUME_CHECKPOINT_DIR}"
    echo "For true resume, use the whole checkpoints/<step> directory, not only pretrained_model."
    exit 1
  fi

  if [[ -z "${RESUME_OUTPUT_DIR}" ]]; then
    RESUME_OUTPUT_DIR="$(dirname "$(dirname "${RESUME_CHECKPOINT_DIR%/}")")"
  fi

  RESUME_CONFIG_JOB_NAME="$(
    python -c 'import json, sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("job_name") or "")' \
      "${RESUME_CONFIG_PATH}"
  )"
  RESUME_CONFIG_OUTPUT_DIR="$(
    python -c 'import json, sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("output_dir") or "")' \
      "${RESUME_CONFIG_PATH}"
  )"

  if [[ -n "${RESUME_CONFIG_OUTPUT_DIR}" && "${RESUME_CONFIG_OUTPUT_DIR}" != "${RESUME_OUTPUT_DIR}" ]]; then
    echo "[warn] RESUME_OUTPUT_DIR=${RESUME_OUTPUT_DIR}"
    echo "[warn] config output_dir=${RESUME_CONFIG_OUTPUT_DIR}"
  fi
fi

if [[ "${LOAD_TEXT_ENCODER}" != "true" && ! -d "${TEXT_EMBED_CACHE_DIR}" ]]; then
  echo "LOAD_TEXT_ENCODER=false but TEXT_EMBED_CACHE_DIR does not exist: ${TEXT_EMBED_CACHE_DIR}"
  echo "Generate a RoboTwin repo list first with:"
  echo "  python tools/discover_robotwin_repos.py --robotwin-root \"${ROBOTWIN_ROOT}\" --output-file outputs/WSA_Large/_repo_id_files/robotwin.txt --require-three-cameras ${ROBOTWIN_REQUIRE_THREE_CAMERAS}"
  echo "Then precompute text embeddings with:"
  echo "  python tools/precompute_text_embeds.py --repo-id-file outputs/WSA_Large/_repo_id_files/robotwin.txt --text-embedding-cache-dir \"${TEXT_EMBED_CACHE_DIR}\" --device cuda"
  echo "Or set LOAD_TEXT_ENCODER=true."
  exit 1
fi

if [[ -n "${NORMALIZATION_STATS_PATH}" && ! -f "${NORMALIZATION_STATS_PATH}" ]]; then
  echo "NORMALIZATION_STATS_PATH does not exist: ${NORMALIZATION_STATS_PATH}"
  exit 1
fi

if [[ -n "${ACTION_STATS_PATH}" && ! -f "${ACTION_STATS_PATH}" ]]; then
  echo "ACTION_STATS_PATH does not exist: ${ACTION_STATS_PATH}"
  exit 1
fi

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating RoboTwin dataset mappings..."
  for ds_dir in "${DATASET_REPO_IDS[@]}"; do
    info_path="${ds_dir}/meta/info.json"
    python -c 'import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
image_keys = json.loads(sys.argv[3])
features = set(info.get("features", {}).keys())
required = {"observation.state", "action"}
required.update(f"observation.images.{key}" for key in image_keys)
print("{} -> robot_type={}, features={}".format(sys.argv[2], info.get("robot_type"), len(features)))
missing = sorted(required - features)
if missing:
    raise SystemExit(f"Missing required WSA_Large RobotWin features for {sys.argv[2]}: {missing}")
' "${info_path}" "${ds_dir}" "${IMAGE_KEYS}"
  done
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

echo "Discovered ${#DATASET_REPO_IDS[@]} RoboTwin datasets under ${ROBOTWIN_ROOT}"
printf '  %s\n' "${DATASET_REPO_IDS[@]}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
BOOTSTRAP_TAG="${BOOTSTRAP_TAG:-2nd-160k-pretrained300k}"
if [[ "${RESUME}" == "true" ]]; then
  JOB_NAME="${JOB_NAME:-${RESUME_CONFIG_JOB_NAME:-${WSA_LARGE_VARIANT}-robotwin-3d-${ACTION_TYPE}-${BOOTSTRAP_TAG}-resume}}"
  OUTPUT_DIR="${RESUME_OUTPUT_DIR}"
  REPO_ID_FILE_DIR="${OUTPUT_DIR}/_repo_id_files"
else
  JOB_NAME="${JOB_NAME:-${WSA_LARGE_VARIANT}-robotwin-3d-${ACTION_TYPE}-${BOOTSTRAP_TAG}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
  REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
fi
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "RESUME=${RESUME}"
if [[ "${RESUME}" == "true" ]]; then
  echo "RESUME_CHECKPOINT_DIR=${RESUME_CHECKPOINT_DIR}"
  echo "RESUME_CONFIG_PATH=${RESUME_CONFIG_PATH}"
  echo "RESUME_OUTPUT_DIR=${RESUME_OUTPUT_DIR}"
fi
echo "WSA_LARGE_VARIANT=${WSA_LARGE_VARIANT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "ACTION_DIM=${ACTION_DIM}, PROPRIO_DIM=${PROPRIO_DIM}"
echo "NORM_DEFAULT_MODE=${NORM_DEFAULT_MODE}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "NORMALIZATION_STATS_PATH=${NORMALIZATION_STATS_PATH:-<auto-compute>}"
echo "ACTION_STATS_PATH=${ACTION_STATS_PATH:-<none>}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}, IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "MASK_ACTION_DIM_PADDING_LOSS=${MASK_ACTION_DIM_PADDING_LOSS}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH}"
echo "SKIP_DIT_LOAD_FROM_PRETRAIN=${SKIP_DIT_LOAD_FROM_PRETRAIN}"
echo "ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH}"
echo "FUTURE_3D_PRETRAINED_PATH=${FUTURE_3D_PRETRAINED_PATH}"
echo "NUM_FRAMES=${NUM_FRAMES}, ACTION_HORIZON=${ACTION_HORIZON}, ACTION_VIDEO_FREQ_RATIO=${ACTION_VIDEO_FREQ_RATIO}"
echo "VIDEO_SIZE=[${VIDEO_HEIGHT},${VIDEO_WIDTH}], CONCAT_MULTI_CAMERA=${CONCAT_MULTI_CAMERA}"
echo "STANDARDIZE_VIDEO_SIZE_BY_CAMERAS=${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
echo "NUM_EPOCHS=${NUM_EPOCHS:-<disabled>}"
echo "TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-<disabled>}"
echo "USE_DIST_LOADING=${USE_DIST_LOADING}"
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
    --dataset.repo_id="multidata_from_file"
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

    --seed="${TRAIN_SEED}"
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --eval_freq="${EVAL_FREQ}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=WSA_Large
    --wandb.mode=${WANDB_MODE}
)

if [[ "${RESUME}" == "true" ]]; then
    ARGS+=(--resume=true --config_path="${RESUME_CONFIG_PATH}")
fi

if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
    ARGS+=(--dataset.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}")
fi
ARGS+=(--dataset.text_embedding_cache_max_entries="${TEXT_EMBED_CACHE_MAX_ENTRIES}")

if [[ -n "${POLICY_INIT_PATH}" ]]; then
    ARGS+=(--policy.pretrained_path="${POLICY_INIT_PATH}")
fi

if [[ -n "${STEPS}" ]]; then
    ARGS+=(--steps="${STEPS}")
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

if [[ -n "${ACTION_STATS_PATH}" ]]; then
    ARGS+=(--policy.action_stats_path="${ACTION_STATS_PATH}")
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
