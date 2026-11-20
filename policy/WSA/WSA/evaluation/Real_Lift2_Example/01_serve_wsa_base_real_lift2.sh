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
INFER_HORIZON="${INFER_HORIZON:-30}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Clear the junk and items off the desktop.}"
RTC_ENABLED="${RTC_ENABLED:-false}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-exp}"
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"
OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE="${OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE:-true}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Please set CHECKPOINT_DIR to a WSA checkpoint step dir or pretrained_model dir."
  exit 1
fi

ARGS=(
  python evaluation/Real_Lift2_Example/model_server.py
  --ckpt_path="${CHECKPOINT_DIR}"
  --host="${HOST}"
  --port="${PORT}"
  --device="${DEVICE}"
  --dtype="${DTYPE}"
  --infer_horizon="${INFER_HORIZON}"
  --resize_size="${RESIZE_SIZE}"
  --default_prompt="${DEFAULT_PROMPT}"
)

if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  ARGS+=(--num_inference_steps="${NUM_INFERENCE_STEPS}")
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

case "${OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE,,}" in
  true|1|yes|y|on)
    ARGS+=(--omit_visual_tokens_in_causal_inference)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-omit_visual_tokens_in_causal_inference)
    ;;
  *)
    echo "Invalid OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE=${OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE}"
    echo "Expected one of: true/false, 1/0, yes/no, on/off"
    exit 1
    ;;
esac

if [[ -n "${STATS_KEY:-}" ]]; then
  ARGS+=(--stats_key="${STATS_KEY}")
fi

if [[ -n "${STATS_PATH:-}" ]]; then
  ARGS+=(--stats_path="${STATS_PATH}")
fi

if [[ -n "${ACTION_MODE:-}" ]]; then
  ARGS+=(--action_mode="${ACTION_MODE}")
fi

if [[ "${RTC_ENABLED,,}" == "true" ]]; then
  ARGS+=(--rtc_enabled)
  ARGS+=(--rtc_execution_horizon="${RTC_EXECUTION_HORIZON}")
  ARGS+=(--rtc_max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}")
  ARGS+=(--rtc_prefix_attention_schedule="${RTC_PREFIX_ATTENTION_SCHEDULE}")
fi

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
