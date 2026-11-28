#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REAL_LIFT2_RUNTIME_ROOT="${REAL_LIFT2_RUNTIME_ROOT:-${HOME}/ROS2_LIFT_Play/act}"
ENTRY_SCRIPT="${PROJ_ROOT}/evaluation/Real_Lift2_Example/main.py"
RUNTIME_MODULE="${PROJ_ROOT}/evaluation/Real_Lift2_Example/runtime.py"
INFERENCE_MODULE="${PROJ_ROOT}/evaluation/Real_Lift2_Example/inference.py"
REMOTE_CLIENT_MODULE="${PROJ_ROOT}/evaluation/Real_Lift2_Example/remote_client.py"
REQUEST_BUILDER_MODULE="${PROJ_ROOT}/evaluation/Real_Lift2_Example/request_builder.py"
WEBSOCKET_CLIENT_MODULE="${PROJ_ROOT}/evaluation/Real_Lift2_Example/websocket_client.py"
MSGPACK_NUMPY_MODULE="${PROJ_ROOT}/evaluation/Real_Lift2_Example/msgpack_numpy.py"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export REAL_LIFT2_RUNTIME_ROOT
export PYTHONPATH="${REAL_LIFT2_RUNTIME_ROOT}:${PROJ_ROOT}/src:${PROJ_ROOT}:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

for required_file in \
  "${ENTRY_SCRIPT}" \
  "${RUNTIME_MODULE}" \
  "${INFERENCE_MODULE}" \
  "${REMOTE_CLIENT_MODULE}" \
  "${REQUEST_BUILDER_MODULE}" \
  "${WEBSOCKET_CLIENT_MODULE}" \
  "${MSGPACK_NUMPY_MODULE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required deployment file is missing:"
    echo "  ${required_file}"
    echo "Please sync the full evaluation/Real_Lift2_Example directory to the target machine."
    exit 1
  fi
done

WS_URL="${WS_URL:-ws://127.0.0.1:8000}"
PROMPT="${PROMPT:-Clear the junk and items off the desktop.}"
FRAME_RATE="${FRAME_RATE:-60}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
SEND_IMAGE_HEIGHT="${SEND_IMAGE_HEIGHT:-}"
SEND_IMAGE_WIDTH="${SEND_IMAGE_WIDTH:-}"
MAX_PUBLISH_STEP="${MAX_PUBLISH_STEP:-10000}"
RECORD_MODE="${RECORD_MODE:-}"
STATE_DIM="${STATE_DIM:-}"
ACTION_DIM="${ACTION_DIM:-}"
INFERENCE_MODE="${INFERENCE_MODE:-}"
PREFETCH_LEAD_STEPS="${PREFETCH_LEAD_STEPS:-}"
LOG_TIMING_EVERY="${LOG_TIMING_EVERY:-}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-}"
RTC_LATENCY_LOOKBACK="${RTC_LATENCY_LOOKBACK:-}"
SEED="${SEED:-}"
SAFE_STOP_BODY_HEIGHT="${SAFE_STOP_BODY_HEIGHT:-}"
SAFE_STOP_PUBLISH_STEPS="${SAFE_STOP_PUBLISH_STEPS:-}"
SAFE_STOP_HOME_ARMS="${SAFE_STOP_HOME_ARMS:-}"
SAFE_STOP_HOME_PUBLISH_STEPS="${SAFE_STOP_HOME_PUBLISH_STEPS:-}"
MANUAL_HOME_PUBLISH_STEPS="${MANUAL_HOME_PUBLISH_STEPS:-}"
MANUAL_HOME_RESUME_GUARD_STEPS="${MANUAL_HOME_RESUME_GUARD_STEPS:-}"

ARGS=(
  python "${ENTRY_SCRIPT}"
  --ws_url="${WS_URL}"
  --prompt="${PROMPT}"
  --frame_rate="${FRAME_RATE}"
  --image_history_interval="${IMAGE_HISTORY_INTERVAL}"
  --max_publish_step="${MAX_PUBLISH_STEP}"
)

if [[ -n "${SEND_IMAGE_HEIGHT}" ]]; then
  ARGS+=(--send_image_height="${SEND_IMAGE_HEIGHT}")
fi

if [[ -n "${SEND_IMAGE_WIDTH}" ]]; then
  ARGS+=(--send_image_width="${SEND_IMAGE_WIDTH}")
fi

if [[ -n "${DATA_CONFIG:-}" ]]; then
  ARGS+=(--data="${DATA_CONFIG}")
fi

if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed="${SEED}")
fi

if [[ "${USE_BASE:-false}" == "true" ]]; then
  ARGS+=(--use_base)
fi

if [[ -n "${FIXED_BODY_HEIGHT:-}" ]]; then
  ARGS+=(--fixed_body_height="${FIXED_BODY_HEIGHT}")
fi

if [[ -n "${GRIPPER_GATE:-}" ]]; then
  ARGS+=(--gripper_gate="${GRIPPER_GATE}")
fi

if [[ -n "${RECORD_MODE}" ]]; then
  ARGS+=(--record="${RECORD_MODE}")
fi

if [[ -n "${STATE_DIM}" ]]; then
  ARGS+=(--state_dim="${STATE_DIM}")
fi

if [[ -n "${ACTION_DIM}" ]]; then
  ARGS+=(--action_dim="${ACTION_DIM}")
fi

if [[ -n "${INFERENCE_MODE}" ]]; then
  ARGS+=(--inference_mode="${INFERENCE_MODE}")
fi

if [[ -n "${PREFETCH_LEAD_STEPS}" ]]; then
  ARGS+=(--prefetch_lead_steps="${PREFETCH_LEAD_STEPS}")
fi

if [[ -n "${LOG_TIMING_EVERY}" ]]; then
  ARGS+=(--log_timing_every="${LOG_TIMING_EVERY}")
fi

if [[ -n "${RTC_EXECUTION_HORIZON}" ]]; then
  ARGS+=(--rtc_execution_horizon="${RTC_EXECUTION_HORIZON}")
fi

if [[ -n "${RTC_MAX_GUIDANCE_WEIGHT}" ]]; then
  ARGS+=(--rtc_max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}")
fi

if [[ -n "${RTC_QUEUE_THRESHOLD}" ]]; then
  ARGS+=(--rtc_queue_threshold="${RTC_QUEUE_THRESHOLD}")
fi

if [[ -n "${RTC_LATENCY_LOOKBACK}" ]]; then
  ARGS+=(--rtc_latency_lookback="${RTC_LATENCY_LOOKBACK}")
fi

if [[ -n "${SAFE_STOP_BODY_HEIGHT}" ]]; then
  ARGS+=(--safe_stop_body_height="${SAFE_STOP_BODY_HEIGHT}")
fi

if [[ -n "${SAFE_STOP_PUBLISH_STEPS}" ]]; then
  ARGS+=(--safe_stop_publish_steps="${SAFE_STOP_PUBLISH_STEPS}")
fi

if [[ "${SAFE_STOP_HOME_ARMS:-false}" == "true" ]]; then
  ARGS+=(--safe_stop_home_arms)
fi

if [[ -n "${SAFE_STOP_HOME_PUBLISH_STEPS}" ]]; then
  ARGS+=(--safe_stop_home_publish_steps="${SAFE_STOP_HOME_PUBLISH_STEPS}")
fi

if [[ -n "${MANUAL_HOME_PUBLISH_STEPS}" ]]; then
  ARGS+=(--manual_home_publish_steps="${MANUAL_HOME_PUBLISH_STEPS}")
fi

if [[ -n "${MANUAL_HOME_RESUME_GUARD_STEPS}" ]]; then
  ARGS+=(--manual_home_resume_guard_steps="${MANUAL_HOME_RESUME_GUARD_STEPS}")
fi

"${ARGS[@]}"
