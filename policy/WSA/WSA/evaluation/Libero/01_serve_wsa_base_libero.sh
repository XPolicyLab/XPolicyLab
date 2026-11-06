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
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
INFER_HORIZON="${INFER_HORIZON:-10}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REQUEST_IMAGE_HEIGHT="${REQUEST_IMAGE_HEIGHT:-256}"
REQUEST_IMAGE_WIDTH="${REQUEST_IMAGE_WIDTH:-256}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Execute the LIBERO task.}"
STATS_KEY="${STATS_KEY:-franka}"
ACTION_MODE="${ACTION_MODE:-abs}"
LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER:-false}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
TEXT_EMBED_CONTEXT_LEN="${TEXT_EMBED_CONTEXT_LEN:-128}"
RTC_ENABLED="${RTC_ENABLED:-false}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-linear}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Please set CHECKPOINT_DIR to a WSA checkpoint step dir or pretrained_model dir."
  exit 1
fi

ARGS=(
  python "${SCRIPT_DIR}/model_server.py"
  --ckpt_path="${CHECKPOINT_DIR}"
  --host="${HOST}"
  --port="${PORT}"
  --device="${DEVICE}"
  --dtype="${DTYPE}"
  --infer_horizon="${INFER_HORIZON}"
  --resize_size="${RESIZE_SIZE}"
  --request_image_height="${REQUEST_IMAGE_HEIGHT}"
  --request_image_width="${REQUEST_IMAGE_WIDTH}"
  --default_prompt="${DEFAULT_PROMPT}"
  --stats_key="${STATS_KEY}"
  --action_mode="${ACTION_MODE}"
  --text_embed_context_len="${TEXT_EMBED_CONTEXT_LEN}"
)

if [[ -n "${STATS_PATH:-}" ]]; then
  ARGS+=(--stats_path="${STATS_PATH}")
fi

if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
  ARGS+=(--text_embed_cache_dir="${TEXT_EMBED_CACHE_DIR}")
fi

case "${LOAD_TEXT_ENCODER,,}" in
  true|1|yes|y|on)
    ARGS+=(--load_text_encoder)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-load_text_encoder)
    ;;
  *)
    echo "Invalid LOAD_TEXT_ENCODER=${LOAD_TEXT_ENCODER}"
    exit 1
    ;;
esac

case "${RTC_ENABLED,,}" in
  true|1|yes|y|on)
    ARGS+=(--rtc_enabled)
    ARGS+=(--rtc_execution_horizon="${RTC_EXECUTION_HORIZON}")
    ARGS+=(--rtc_max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}")
    ARGS+=(--rtc_prefix_attention_schedule="${RTC_PREFIX_ATTENTION_SCHEDULE}")
    ;;
  false|0|no|n|off)
    ;;
  *)
    echo "Invalid RTC_ENABLED=${RTC_ENABLED}"
    exit 1
    ;;
esac

if [[ -n "${LOAD_DEVICE:-}" ]]; then
  ARGS+=(--load_device="${LOAD_DEVICE}")
fi

if [[ -n "${COSMOS_DEVICE:-}" ]]; then
  ARGS+=(--cosmos_device="${COSMOS_DEVICE}")
fi

if [[ -n "${QWEN3_VL_PROCESSOR_PATH:-}" ]]; then
  ARGS+=(--qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}")
fi

if [[ -n "${QWEN3_VL_PRETRAINED_PATH:-}" ]]; then
  ARGS+=(--qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}")
fi

if [[ -n "${COSMOS_TOKENIZER_PATH_OR_NAME:-}" ]]; then
  ARGS+=(--cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}")
fi

if [[ -n "${DA3_MODEL_PATH_OR_NAME:-}" ]]; then
  ARGS+=(--da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}")
fi

if [[ -n "${DA3_CODE_ROOT:-}" ]]; then
  ARGS+=(--da3_code_root="${DA3_CODE_ROOT}")
fi

"${ARGS[@]}"
