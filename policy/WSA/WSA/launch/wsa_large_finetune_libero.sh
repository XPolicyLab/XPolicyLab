#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6389}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=${WANDB_MODE:-offline}
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

LIBERO_ROOT="${LIBERO_ROOT:-/path/to/libero-lerobot-v30}"
if [[ "${LOAD_TEXT_ENCODER}" == "true" ]]; then
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
else
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJ_ROOT}/outputs/WSA_Large/text_embeds/libero}"
fi
TEXT_EMBED_CACHE_MAX_ENTRIES="${TEXT_EMBED_CACHE_MAX_ENTRIES:-0}"
NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH:-outputs/norm_stats/libero_all_chunk10/franka/abs/stats.json}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
USE_DIST_LOADING="${USE_DIST_LOADING:-false}"

NUM_FRAMES="${NUM_FRAMES:-13}"
ACTION_DIM="${ACTION_DIM:-24}"
PROPRIO_DIM="${PROPRIO_DIM:-24}"
ACTION_HORIZON="$((NUM_FRAMES - 1))"
N_ACTION_STEPS=5
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-3}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-224}"
VIDEO_WIDTH="${VIDEO_WIDTH:-448}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-horizontal}"
STANDARDIZE_VIDEO_SIZE_BY_CAMERAS="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS:-true}"
NORM_DEFAULT_MODE="${NORM_DEFAULT_MODE:-z-score}"
ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"
IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-110000}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-200}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-2}"
NUM_WORKERS="${NUM_WORKERS:-12}"

LR="${LR:-7.5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DECAY_LR="${DECAY_LR:-5.0e-6}"
LAMBDA_VIDEO="${LAMBDA_VIDEO:-0.1}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
LAMBDA_3D="${LAMBDA_3D:-0.1}"
# Match the Real_Piper dual-view recipe: 2 views, 216 tokens/view, horizontal layout.
DA3_NUM_VIEWS="${DA3_NUM_VIEWS:-2}"
PROCESSOR_NUM_OUTPUT_CAMERAS="${PROCESSOR_NUM_OUTPUT_CAMERAS:-${DA3_NUM_VIEWS}}"
FUTURE_3D_TOKENS_PER_VIEW="${FUTURE_3D_TOKENS_PER_VIEW:-216}"
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
FUTURE_3D_TARGET_INDEX="${FUTURE_3D_TARGET_INDEX:--1}"
DTYPE="${DTYPE:-bfloat16}"
WSA_LARGE_CHECKPOINT_MIXED_ATTN="${WSA_LARGE_CHECKPOINT_MIXED_ATTN:-false}"
ACTION_MODE="${ACTION_MODE:-abs}"

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

case "${SKIP_DIT_LOAD_FROM_PRETRAIN}" in
  true|false)
    ;;
  *)
    echo "Unsupported SKIP_DIT_LOAD_FROM_PRETRAIN=${SKIP_DIT_LOAD_FROM_PRETRAIN}. Expected true or false."
    exit 1
    ;;
esac

case "${ACTION_MODE}" in
  abs)
    ;;
  *)
    echo "Unsupported ACTION_MODE=${ACTION_MODE}. LIBERO finetuning expects abs."
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
        ds_name="$(basename "${ds_dir}")"
        case "${ds_name}" in
          libero_*_lerobot_v30)
            if [[ -d "${ds_dir}/data" || -d "${ds_dir}/videos" ]]; then
              echo "${ds_dir}"
            fi
            ;;
        esac
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${LIBERO_ROOT}")

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No LIBERO v3.0 datasets found under LIBERO_ROOT=${LIBERO_ROOT}"
  echo "Expected directories like libero_goal_no_noops_1.0.0_lerobot_v30"
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

if [[ "${LOAD_TEXT_ENCODER}" != "true" && ! -d "${TEXT_EMBED_CACHE_DIR}" ]]; then
  echo "LOAD_TEXT_ENCODER=false but TEXT_EMBED_CACHE_DIR does not exist: ${TEXT_EMBED_CACHE_DIR}"
  echo "Precompute text embeddings with:"
  echo "  python tools/precompute_text_embeds.py --repo-id-file <repo_id_file.txt> --text-embedding-cache-dir \"${TEXT_EMBED_CACHE_DIR}\" --device cuda"
  echo "Or set LOAD_TEXT_ENCODER=true."
  exit 1
fi

if [[ -n "${NORMALIZATION_STATS_PATH}" && ! -f "${NORMALIZATION_STATS_PATH}" ]]; then
  echo "NORMALIZATION_STATS_PATH does not exist: ${NORMALIZATION_STATS_PATH}"
  exit 1
fi

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating LIBERO dataset mappings..."
  for ds_dir in "${DATASET_REPO_IDS[@]}"; do
    info_path="${ds_dir}/meta/info.json"
    python -c 'import json, sys
from lerobot.transforms.constants import infer_embodiment_variant
info = json.load(open(sys.argv[1], encoding="utf-8"))
robot_type = info["robot_type"]
resolved = infer_embodiment_variant(robot_type, info.get("features", {}))
codebase_version = info.get("codebase_version", "unknown")
print(f"{sys.argv[2]} -> codebase={codebase_version}, robot_type={robot_type}, resolved={resolved}")
if codebase_version != "v3.0":
    raise SystemExit(f"Dataset is not v3.0: {sys.argv[2]}")
if resolved != "libero_franka":
    raise SystemExit(f"Unexpected mapping resolution for {sys.argv[2]}: {resolved}")
' "${info_path}" "${ds_dir}"
  done
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

echo "Discovered ${#DATASET_REPO_IDS[@]} LIBERO datasets under ${LIBERO_ROOT}"
printf '  %s\n' "${DATASET_REPO_IDS[@]}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
JOB_NAME="${WSA_LARGE_VARIANT}-pretrained-libero4-$(date +'%Y_%m_%d_%H_%M_%S')"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "WSA_LARGE_VARIANT=${WSA_LARGE_VARIANT}"
echo "ACTION_DIM=${ACTION_DIM}, PROPRIO_DIM=${PROPRIO_DIM}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH:-<none>}"
echo "SKIP_DIT_LOAD_FROM_PRETRAIN=${SKIP_DIT_LOAD_FROM_PRETRAIN}"
echo "ACTION_MODE=${ACTION_MODE}"
echo "ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH}"
echo "FUTURE_3D_PRETRAINED_PATH=${FUTURE_3D_PRETRAINED_PATH}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "ACTION_HORIZON=${ACTION_HORIZON}"
echo "ACTION_VIDEO_FREQ_RATIO=${ACTION_VIDEO_FREQ_RATIO}"
echo "CONCAT_MULTI_CAMERA=${CONCAT_MULTI_CAMERA}"
echo "STANDARDIZE_VIDEO_SIZE_BY_CAMERAS=${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
echo "NORM_DEFAULT_MODE=${NORM_DEFAULT_MODE}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}, IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "STEPS=${STEPS}"
echo "NUM_EPOCHS=${NUM_EPOCHS:-<disabled>}"
echo "TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-<disabled>}"
echo "EVAL_FREQ=${EVAL_FREQ}, EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES}"
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
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_teacher_process_res="${DA3_TEACHER_PROCESS_RES}"
    --policy.action_norm_default_mode="${NORM_DEFAULT_MODE}"
    --policy.optimizer_lr="${LR}"
    --policy.optimizer_weight_decay="${WEIGHT_DECAY}"

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.action_mode="${ACTION_MODE}"
    --dataset.num_frames="${NUM_FRAMES}"
    --dataset.action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO}"
    --dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]"
    --dataset.standardize_video_size_by_cameras="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
    --dataset.context_len=128
    --dataset.val_set_proportion=0.0
    --dataset.skip_padding_as_possible=false
    --dataset.concat_multi_camera="${CONCAT_MULTI_CAMERA}"
    --dataset.processor_num_output_cameras="${PROCESSOR_NUM_OUTPUT_CAMERAS}"
    --dataset.processor_action_output_dim="${ACTION_DIM}"
    --dataset.processor_proprio_output_dim="${PROPRIO_DIM}"
    --dataset.processor_norm_default_mode="${NORM_DEFAULT_MODE}"
    --dataset.future_3d_target_index="${FUTURE_3D_TARGET_INDEX}"

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --steps="${STEPS}"
    --eval_freq="${EVAL_FREQ}"
    --eval_max_batches="${EVAL_MAX_BATCHES}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=WSA_Large
    --wandb.mode=${WANDB_MODE}
)

if [[ -n "${POLICY_INIT_PATH}" ]]; then
    ARGS+=(--policy.pretrained_path="${POLICY_INIT_PATH}")
fi

if [[ -n "${NUM_EPOCHS}" ]]; then
    ARGS+=(--policy.train_num_epochs="${NUM_EPOCHS}")
fi

if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
    ARGS+=(--dataset.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}")
fi
ARGS+=(--dataset.text_embedding_cache_max_entries="${TEXT_EMBED_CACHE_MAX_ENTRIES}")

if [[ -n "${WARMUP_STEPS}" ]]; then
    ARGS+=(--policy.scheduler_warmup_steps="${WARMUP_STEPS}")
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
    echo "USE_DIST_LOADING=true is not supported for WSA_Large in this framework."
    echo "Leave USE_DIST_LOADING=false so Accelerate can shard the dataloader correctly."
    exit 1
fi

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"
    )
fi

accelerate launch "${ARGS[@]}"
