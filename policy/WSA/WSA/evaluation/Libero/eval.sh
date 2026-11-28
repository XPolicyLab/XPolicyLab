#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIBERO_HOME="${LIBERO_HOME:-}"
if [[ -n "${LIBERO_HOME}" ]]; then
  export PYTHONPATH="${LIBERO_HOME}:${PYTHONPATH:-}"
elif [[ -d "${PROJ_ROOT}/LIBERO" ]]; then
  export PYTHONPATH="${PROJ_ROOT}/LIBERO:${PYTHONPATH:-}"
elif [[ -d "${PROJ_ROOT}/third_party/LIBERO" ]]; then
  export PYTHONPATH="${PROJ_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
fi

cd "${PROJ_ROOT}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
TASK_ID="${TASK_ID:-}"
SEED="${SEED:-7}"
STATS_KEY="${STATS_KEY:-franka}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
INFER_HORIZON="${INFER_HORIZON:-}"
WS_URL="${WS_URL:-}"
MODE_TAG="local"
if [[ -n "${WS_URL}" ]]; then
  MODE_TAG="split_ws"
fi
VIDEO_ROOT="${VIDEO_ROOT:-${PROJ_ROOT}/evaluation/Libero/output}"
VIDEO_DIR="${VIDEO_DIR:-${VIDEO_ROOT}/${TASK_SUITE_NAME}}"

QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"

ARGS=(
  --args.task_suite_name "${TASK_SUITE_NAME}"
  --args.seed "${SEED}"
  --args.stats_key "${STATS_KEY}"
  --args.num_trials_per_task "${NUM_TRIALS_PER_TASK}"
  --args.video_dir "${VIDEO_DIR}"
)

if [[ -n "${PRETRAINED_CKPT}" ]]; then
  ARGS+=(--args.ckpt_path "${PRETRAINED_CKPT}")
fi

if [[ -n "${WS_URL}" ]]; then
  ARGS+=(--args.ws_url "${WS_URL}")
fi

case "${DISABLE_3D_TEACHER_FOR_EVAL,,}" in
  true|1|yes|y|on)
    ARGS+=(--args.disable_3d_teacher_for_eval)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-args.disable_3d_teacher_for_eval)
    ;;
  *)
    echo "Invalid DISABLE_3D_TEACHER_FOR_EVAL=${DISABLE_3D_TEACHER_FOR_EVAL}"
    echo "Expected one of: true/false, 1/0, yes/no, on/off"
    exit 1
    ;;
esac

if [[ -n "${TASK_ID}" ]]; then
  ARGS+=(--args.task_id "${TASK_ID}")
fi

if [[ -n "${INFER_HORIZON}" ]]; then
  ARGS+=(--args.infer_horizon "${INFER_HORIZON}")
fi

if [[ -n "${QWEN3_VL_PRETRAINED_PATH}" ]]; then
  ARGS+=(--args.qwen3_vl_pretrained_path "${QWEN3_VL_PRETRAINED_PATH}")
fi

if [[ -n "${QWEN3_VL_PROCESSOR_PATH}" ]]; then
  ARGS+=(--args.qwen3_vl_processor_path "${QWEN3_VL_PROCESSOR_PATH}")
fi

if [[ -n "${COSMOS_TOKENIZER_PATH_OR_NAME}" ]]; then
  ARGS+=(--args.cosmos_tokenizer_path_or_name "${COSMOS_TOKENIZER_PATH_OR_NAME}")
fi

if [[ -n "${DA3_MODEL_PATH_OR_NAME}" ]]; then
  ARGS+=(--args.da3_model_path_or_name "${DA3_MODEL_PATH_OR_NAME}")
fi

if [[ -n "${DA3_CODE_ROOT}" ]]; then
  ARGS+=(--args.da3_code_root "${DA3_CODE_ROOT}")
fi

if [[ -z "${PRETRAINED_CKPT}" && -z "${WS_URL}" ]]; then
  echo "Please set either PRETRAINED_CKPT for local evaluation or WS_URL for split websocket policy serving."
  exit 1
fi

echo "LIBERO task suite: ${TASK_SUITE_NAME}"
echo "LIBERO task id   : ${TASK_ID:-all}"
echo "Eval mode        : ${MODE_TAG}"
echo "Output root      : ${VIDEO_ROOT}"
echo "Output dir       : ${VIDEO_DIR}"

python evaluation/Libero/inference.py "${ARGS[@]}"
