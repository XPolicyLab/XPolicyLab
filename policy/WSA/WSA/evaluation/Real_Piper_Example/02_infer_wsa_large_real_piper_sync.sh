#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

WS_HOST="${WS_HOST:-10.60.43.33}"
WS_PORT="${WS_PORT:-8102}"
TASK_PROMPT="${TASK_PROMPT:-Sort desktop objects and place them in designated locations.}"
PUBLISH_RATE="${PUBLISH_RATE:-15}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_STEPS="${MAX_STEPS:-10000}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
SYNC_TOLERANCE="${SYNC_TOLERANCE:-0.1}"
SEND_IMAGE_HEIGHT="${SEND_IMAGE_HEIGHT:-}"
SEND_IMAGE_WIDTH="${SEND_IMAGE_WIDTH:-}"

FRONT_CAM_TOPIC="${FRONT_CAM_TOPIC:-/ob_camera_02/color/image_raw}"
WRIST_CAM_TOPIC="${WRIST_CAM_TOPIC:-/ob_camera_01/color/image_raw}"
JOINT_STATE_TOPIC="${JOINT_STATE_TOPIC:-joint_states_single}"
JOINT_CMD_TOPIC="${JOINT_CMD_TOPIC:-js_cmd}"

FIRST_INFERENCE_CHECK="${FIRST_INFERENCE_CHECK:-false}"
START_PROMPT="${START_PROMPT:-true}"
GRIPPER_POSTPROCESS="${GRIPPER_POSTPROCESS:-true}"
JPEG_ROUNDTRIP="${JPEG_ROUNDTRIP:-true}"
IMAGE_COLOR_MODE="${IMAGE_COLOR_MODE:-auto}"
EXPECTED_STATS_KEY="${EXPECTED_STATS_KEY:-real_piper}"
ALLOW_STATS_KEY_MISMATCH="${ALLOW_STATS_KEY_MISMATCH:-false}"
ALLOW_ACTION_DIM_MISMATCH="${ALLOW_ACTION_DIM_MISMATCH:-false}"
LOG_TAG="${LOG_TAG:-WSA_Large-Piper}"
MANUAL_RESET="${MANUAL_RESET:-true}"
MANUAL_RESET_RESUME_HOLD_STEPS="${MANUAL_RESET_RESUME_HOLD_STEPS:-5}"
MANUAL_RESET_REMINDER_INTERVAL="${MANUAL_RESET_REMINDER_INTERVAL:-1.5}"
INIT_TIMEOUT="${INIT_TIMEOUT:-10.0}"
INIT_POSITION_THRESHOLD="${INIT_POSITION_THRESHOLD:-500.0}"

ARGS=(
  python evaluation/Real_Piper_Example/inference_piper_sync.py
  --ws_host="${WS_HOST}"
  --ws_port="${WS_PORT}"
  --task_prompt="${TASK_PROMPT}"
  --publish_rate="${PUBLISH_RATE}"
  --action_horizon="${ACTION_HORIZON}"
  --max_steps="${MAX_STEPS}"
  --image_history_interval="${IMAGE_HISTORY_INTERVAL}"
  --sync_tolerance="${SYNC_TOLERANCE}"
  --front_cam_topic="${FRONT_CAM_TOPIC}"
  --wrist_cam_topic="${WRIST_CAM_TOPIC}"
  --joint_state_topic="${JOINT_STATE_TOPIC}"
  --joint_cmd_topic="${JOINT_CMD_TOPIC}"
  --image_color_mode="${IMAGE_COLOR_MODE}"
  --expected_stats_key="${EXPECTED_STATS_KEY}"
  --log_tag="${LOG_TAG}"
  --manual_reset_resume_hold_steps="${MANUAL_RESET_RESUME_HOLD_STEPS}"
  --manual_reset_reminder_interval="${MANUAL_RESET_REMINDER_INTERVAL}"
  --init_timeout="${INIT_TIMEOUT}"
  --init_position_threshold="${INIT_POSITION_THRESHOLD}"
)

if [[ -n "${SEND_IMAGE_HEIGHT}" ]]; then
  ARGS+=(--send_image_height="${SEND_IMAGE_HEIGHT}")
fi

if [[ -n "${SEND_IMAGE_WIDTH}" ]]; then
  ARGS+=(--send_image_width="${SEND_IMAGE_WIDTH}")
fi

case "${FIRST_INFERENCE_CHECK,,}" in
  true|1|yes|y|on)
    ARGS+=(--first_inference_check)
    ;;
  false|0|no|n|off)
    ;;
  *)
    echo "Invalid FIRST_INFERENCE_CHECK=${FIRST_INFERENCE_CHECK}"
    exit 1
    ;;
esac

case "${START_PROMPT,,}" in
  true|1|yes|y|on)
    ARGS+=(--start_prompt)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-start_prompt)
    ;;
  *)
    echo "Invalid START_PROMPT=${START_PROMPT}"
    exit 1
    ;;
esac

case "${GRIPPER_POSTPROCESS,,}" in
  true|1|yes|y|on)
    ARGS+=(--gripper_postprocess)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-gripper_postprocess)
    ;;
  *)
    echo "Invalid GRIPPER_POSTPROCESS=${GRIPPER_POSTPROCESS}"
    exit 1
    ;;
esac

case "${JPEG_ROUNDTRIP,,}" in
  true|1|yes|y|on)
    ARGS+=(--jpeg_roundtrip)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-jpeg_roundtrip)
    ;;
  *)
    echo "Invalid JPEG_ROUNDTRIP=${JPEG_ROUNDTRIP}"
    exit 1
    ;;
esac

case "${MANUAL_RESET,,}" in
  true|1|yes|y|on)
    ARGS+=(--manual_reset)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-manual_reset)
    ;;
  *)
    echo "Invalid MANUAL_RESET=${MANUAL_RESET}"
    exit 1
    ;;
esac

case "${ALLOW_STATS_KEY_MISMATCH,,}" in
  true|1|yes|y|on)
    ARGS+=(--allow_stats_key_mismatch)
    ;;
  false|0|no|n|off)
    ;;
  *)
    echo "Invalid ALLOW_STATS_KEY_MISMATCH=${ALLOW_STATS_KEY_MISMATCH}"
    exit 1
    ;;
esac

case "${ALLOW_ACTION_DIM_MISMATCH,,}" in
  true|1|yes|y|on)
    ARGS+=(--allow_action_dim_mismatch)
    ;;
  false|0|no|n|off)
    ;;
  *)
    echo "Invalid ALLOW_ACTION_DIM_MISMATCH=${ALLOW_ACTION_DIM_MISMATCH}"
    exit 1
    ;;
esac

if [[ -n "${INIT_JOINT_POSITION:-}" ]]; then
  read -r -a INIT_JOINTS <<< "${INIT_JOINT_POSITION}"
  ARGS+=(--init_joint_position "${INIT_JOINTS[@]}")
fi

if [[ "${INIT_WAIT:-false}" =~ ^(true|1|yes|y|on)$ ]]; then
  ARGS+=(--init_wait)
fi

"${ARGS[@]}"
