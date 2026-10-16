#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8102}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
INFER_HORIZON="${INFER_HORIZON:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Sort desktop objects and place them in designated locations.}"
STATS_KEY="${STATS_KEY:-real_piper}"
STATS_PATH="${STATS_PATH:-}"
ACTION_MODE="${ACTION_MODE:-}"

RTC_ENABLED="${RTC_ENABLED:-false}"
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"

WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
WSA_LARGE_ASSET_ROOT="${WSA_LARGE_ASSET_ROOT:-${PROJ_ROOT}/checkpoints/wsa_large}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${WSA_LARGE_ASSET_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
FUTURE_3D_PRETRAINED_PATH="${FUTURE_3D_PRETRAINED_PATH:-${WSA_LARGE_ASSET_ROOT}/Future3DExpert_linear_interp_Wan22_alphascale_768hdim.pt}"
WSA_LARGE_LOAD_TEXT_ENCODER="${WSA_LARGE_LOAD_TEXT_ENCODER:-true}"
WSA_LARGE_REDIRECT_COMMON_FILES="${WSA_LARGE_REDIRECT_COMMON_FILES:-true}"
WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN="${WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN:-true}"
WSA_LARGE_STATE_KEY="${WSA_LARGE_STATE_KEY:-default}"
WSA_LARGE_VIDEO_HEIGHT="${WSA_LARGE_VIDEO_HEIGHT:-224}"
WSA_LARGE_VIDEO_WIDTH="${WSA_LARGE_VIDEO_WIDTH:-448}"
WSA_LARGE_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS="${WSA_LARGE_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS:-true}"
WSA_LARGE_CONCAT_MULTI_CAMERA="${WSA_LARGE_CONCAT_MULTI_CAMERA:-horizontal}"
WSA_LARGE_TEXT_EMBED_CACHE_DIR="${WSA_LARGE_TEXT_EMBED_CACHE_DIR:-${TEXT_EMBED_CACHE_DIR:-}}"
WSA_LARGE_CONTEXT_LEN="${WSA_LARGE_CONTEXT_LEN:-}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Please set CHECKPOINT_DIR to a WSA_Large real_piper checkpoint step dir or pretrained_model dir."
  exit 1
fi

case "${RTC_ENABLED,,}" in
  true|1|yes|y|on)
    echo "WSA_Large Real_Piper RTC serving is not implemented. Use sync mode with RTC_ENABLED=false."
    exit 1
    ;;
  false|0|no|n|off)
    ;;
  *)
    echo "Invalid RTC_ENABLED=${RTC_ENABLED}"
    echo "Expected one of: true/false, 1/0, yes/no, on/off"
    exit 1
    ;;
esac

ARGS=(
  python evaluation/Real_Piper_Example/model_server_sync.py
  --ckpt_path="${CHECKPOINT_DIR}"
  --host="${HOST}"
  --port="${PORT}"
  --device="${DEVICE}"
  --dtype="${DTYPE}"
  --infer_horizon="${INFER_HORIZON}"
  --default_prompt="${DEFAULT_PROMPT}"
  --stats_key="${STATS_KEY}"
  --wsa_large_model_id="${WAN_MODEL_ID}"
  --wsa_large_tokenizer_model_id="${WAN_TOKENIZER_MODEL_ID}"
  --wsa_large_action_dit_pretrained_path="${ACTION_DIT_PRETRAINED_PATH}"
  --wsa_large_future_3d_pretrained_path="${FUTURE_3D_PRETRAINED_PATH}"
  --wsa_large_state_key="${WSA_LARGE_STATE_KEY}"
  --wsa_large_video_height="${WSA_LARGE_VIDEO_HEIGHT}"
  --wsa_large_video_width="${WSA_LARGE_VIDEO_WIDTH}"
  --wsa_large_concat_multi_camera="${WSA_LARGE_CONCAT_MULTI_CAMERA}"
)

if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  ARGS+=(--num_inference_steps="${NUM_INFERENCE_STEPS}")
fi

if [[ -n "${STATS_PATH}" ]]; then
  ARGS+=(--stats_path="${STATS_PATH}")
fi

if [[ -n "${ACTION_MODE}" ]]; then
  ARGS+=(--action_mode="${ACTION_MODE}")
fi

if [[ -n "${WSA_LARGE_TEXT_EMBED_CACHE_DIR}" ]]; then
  ARGS+=(--wsa_large_text_embedding_cache_dir="${WSA_LARGE_TEXT_EMBED_CACHE_DIR}")
fi

if [[ -n "${WSA_LARGE_CONTEXT_LEN}" ]]; then
  ARGS+=(--wsa_large_context_len="${WSA_LARGE_CONTEXT_LEN}")
fi

case "${DISABLE_3D_TEACHER_FOR_EVAL,,}" in
  true|1|yes|y|on)
    ARGS+=(--disable_3d_teacher_for_eval)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-disable_3d_teacher_for_eval)
    ;;
  *)
    echo "Invalid DISABLE_3D_TEACHER_FOR_EVAL=${DISABLE_3D_TEACHER_FOR_EVAL}"
    echo "Expected one of: true/false, 1/0, yes/no, on/off"
    exit 1
    ;;
esac

case "${WSA_LARGE_LOAD_TEXT_ENCODER,,}" in
  true|1|yes|y|on)
    ARGS+=(--wsa_large_load_text_encoder)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-wsa_large_load_text_encoder)
    ;;
  *)
    echo "Invalid WSA_LARGE_LOAD_TEXT_ENCODER=${WSA_LARGE_LOAD_TEXT_ENCODER}"
    exit 1
    ;;
esac

case "${WSA_LARGE_REDIRECT_COMMON_FILES,,}" in
  true|1|yes|y|on)
    ARGS+=(--wsa_large_redirect_common_files)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-wsa_large_redirect_common_files)
    ;;
  *)
    echo "Invalid WSA_LARGE_REDIRECT_COMMON_FILES=${WSA_LARGE_REDIRECT_COMMON_FILES}"
    exit 1
    ;;
esac

case "${WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN,,}" in
  true|1|yes|y|on)
    ARGS+=(--wsa_large_skip_dit_load_from_pretrain)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-wsa_large_skip_dit_load_from_pretrain)
    ;;
  *)
    echo "Invalid WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN=${WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN}"
    exit 1
    ;;
esac

case "${WSA_LARGE_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS,,}" in
  true|1|yes|y|on)
    ARGS+=(--wsa_large_standardize_video_size_by_cameras)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-wsa_large_standardize_video_size_by_cameras)
    ;;
  *)
    echo "Invalid WSA_LARGE_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS=${WSA_LARGE_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
    exit 1
    ;;
esac

if [[ -n "${LOAD_DEVICE:-}" ]]; then
  ARGS+=(--load_device="${LOAD_DEVICE}")
fi

if [[ -n "${DA3_MODEL_PATH_OR_NAME:-}" ]]; then
  ARGS+=(--da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}")
fi

if [[ -n "${DA3_CODE_ROOT:-}" ]]; then
  ARGS+=(--da3_code_root="${DA3_CODE_ROOT}")
fi

"${ARGS[@]}"
