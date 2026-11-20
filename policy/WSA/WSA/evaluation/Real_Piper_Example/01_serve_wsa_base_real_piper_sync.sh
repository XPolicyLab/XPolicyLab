#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}:${PROJ_ROOT}/evaluation/Real_Lift2_Example:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
INFER_HORIZON="${INFER_HORIZON:-50}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Sort desktop objects and place them in designated locations.}"
STATS_KEY="${STATS_KEY:-real_piper}"
ACTION_MODE="${ACTION_MODE:-abs}"
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"
OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE="${OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE:-true}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Please set CHECKPOINT_DIR to a WSA checkpoint step dir or pretrained_model dir."
  exit 1
fi

ARGS=(
  python "${SCRIPT_DIR}/model_server_sync.py"
  --ckpt_path="${CHECKPOINT_DIR}"
  --host="${HOST}"
  --port="${PORT}"
  --device="${DEVICE}"
  --dtype="${DTYPE}"
  --infer_horizon="${INFER_HORIZON}"
  --resize_size="${RESIZE_SIZE}"
  --default_prompt="${DEFAULT_PROMPT}"
  --stats_key="${STATS_KEY}"
  --action_mode="${ACTION_MODE}"
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
    exit 1
    ;;
esac

if [[ -n "${STATS_PATH:-}" ]]; then
  ARGS+=(--stats_path="${STATS_PATH}")
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
