#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}:${PROJ_ROOT}/evaluation/Real_Lift2_Example:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

WS_HOST="${WS_HOST:-127.0.0.1}"
WS_PORT="${WS_PORT:-8000}"
TASK_PROMPT="${TASK_PROMPT:-Sort desktop objects and place them in designated locations.}"
PUBLISH_RATE="${PUBLISH_RATE:-15}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
MAX_STEPS="${MAX_STEPS:-10000}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
SYNC_TOLERANCE="${SYNC_TOLERANCE:-0.1}"
FRONT_CAM_TOPIC="${FRONT_CAM_TOPIC:-/ob_camera_02/color/image_raw}"
WRIST_CAM_TOPIC="${WRIST_CAM_TOPIC:-/ob_camera_01/color/image_raw}"
JOINT_STATE_TOPIC="${JOINT_STATE_TOPIC:-joint_states_single}"
JOINT_CMD_TOPIC="${JOINT_CMD_TOPIC:-js_cmd}"
IMAGE_COLOR_MODE="${IMAGE_COLOR_MODE:-auto}"
EXPECTED_STATS_KEY="${EXPECTED_STATS_KEY:-real_piper}"
LOG_TAG="${LOG_TAG:-WSA-Piper}"

ARGS=(
  python "${SCRIPT_DIR}/inference_piper_sync.py"
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
)

if [[ -n "${SEND_IMAGE_HEIGHT:-}" ]]; then
  ARGS+=(--send_image_height="${SEND_IMAGE_HEIGHT}")
fi

if [[ -n "${SEND_IMAGE_WIDTH:-}" ]]; then
  ARGS+=(--send_image_width="${SEND_IMAGE_WIDTH}")
fi

if [[ -n "${JOINT_NAMES:-}" ]]; then
  read -r -a JOINT_NAME_ARGS <<< "${JOINT_NAMES}"
  if [[ ${#JOINT_NAME_ARGS[@]} -ne 7 ]]; then
    echo "JOINT_NAMES must contain 7 space-separated names."
    exit 1
  fi
  ARGS+=(--joint_names "${JOINT_NAME_ARGS[@]}")
fi

if [[ -n "${INIT_JOINT_POSITION:-}" ]]; then
  read -r -a INIT_POS_ARGS <<< "${INIT_JOINT_POSITION}"
  if [[ ${#INIT_POS_ARGS[@]} -ne 7 ]]; then
    echo "INIT_JOINT_POSITION must contain 7 space-separated numbers."
    exit 1
  fi
  ARGS+=(--init_joint_position "${INIT_POS_ARGS[@]}")
fi

if [[ -n "${INIT_TIMEOUT:-}" ]]; then
  ARGS+=(--init_timeout="${INIT_TIMEOUT}")
fi

if [[ -n "${INIT_POSITION_THRESHOLD:-}" ]]; then
  ARGS+=(--init_position_threshold="${INIT_POSITION_THRESHOLD}")
fi

if [[ -n "${MANUAL_RESET_RESUME_HOLD_STEPS:-}" ]]; then
  ARGS+=(--manual_reset_resume_hold_steps="${MANUAL_RESET_RESUME_HOLD_STEPS}")
fi

if [[ -n "${MANUAL_RESET_REMINDER_INTERVAL:-}" ]]; then
  ARGS+=(--manual_reset_reminder_interval="${MANUAL_RESET_REMINDER_INTERVAL}")
fi

if [[ -n "${GRIPPER_CLOSE_THRESHOLD:-}" ]]; then
  ARGS+=(--gripper_close_threshold="${GRIPPER_CLOSE_THRESHOLD}")
fi

if [[ -n "${GRIPPER_OPEN_THRESHOLD:-}" ]]; then
  ARGS+=(--gripper_open_threshold="${GRIPPER_OPEN_THRESHOLD}")
fi

if [[ -n "${GRIPPER_CLOSE_OFFSET:-}" ]]; then
  ARGS+=(--gripper_close_offset="${GRIPPER_CLOSE_OFFSET}")
fi

if [[ -n "${GRIPPER_OPEN_OFFSET:-}" ]]; then
  ARGS+=(--gripper_open_offset="${GRIPPER_OPEN_OFFSET}")
fi

if [[ -n "${GRIPPER_MIN:-}" ]]; then
  ARGS+=(--gripper_min="${GRIPPER_MIN}")
fi

if [[ -n "${GRIPPER_MAX:-}" ]]; then
  ARGS+=(--gripper_max="${GRIPPER_MAX}")
fi

bool_arg() {
  local env_value="$1"
  local flag="$2"
  local no_flag="$3"
  case "${env_value,,}" in
    true|1|yes|y|on)
      ARGS+=("${flag}")
      ;;
    false|0|no|n|off)
      ARGS+=("${no_flag}")
      ;;
    *)
      echo "Invalid boolean value ${env_value} for ${flag}"
      exit 1
      ;;
  esac
}

if [[ -n "${FIRST_INFERENCE_CHECK:-}" ]]; then
  case "${FIRST_INFERENCE_CHECK,,}" in
    true|1|yes|y|on) ARGS+=(--first_inference_check) ;;
    false|0|no|n|off) ;;
    *) echo "Invalid FIRST_INFERENCE_CHECK=${FIRST_INFERENCE_CHECK}"; exit 1 ;;
  esac
fi

if [[ -n "${JPEG_ROUNDTRIP:-}" ]]; then
  bool_arg "${JPEG_ROUNDTRIP}" --jpeg_roundtrip --no-jpeg_roundtrip
fi

if [[ -n "${START_PROMPT:-}" ]]; then
  bool_arg "${START_PROMPT}" --start_prompt --no-start_prompt
fi

if [[ -n "${GRIPPER_POSTPROCESS:-}" ]]; then
  bool_arg "${GRIPPER_POSTPROCESS}" --gripper_postprocess --no-gripper_postprocess
fi

if [[ -n "${MANUAL_RESET:-}" ]]; then
  bool_arg "${MANUAL_RESET}" --manual_reset --no-manual_reset
fi

if [[ -n "${INIT_WAIT:-}" ]]; then
  case "${INIT_WAIT,,}" in
    true|1|yes|y|on) ARGS+=(--init_wait) ;;
    false|0|no|n|off) ;;
    *) echo "Invalid INIT_WAIT=${INIT_WAIT}"; exit 1 ;;
  esac
fi

if [[ -n "${ALLOW_STATS_KEY_MISMATCH:-}" ]]; then
  case "${ALLOW_STATS_KEY_MISMATCH,,}" in
    true|1|yes|y|on) ARGS+=(--allow_stats_key_mismatch) ;;
    false|0|no|n|off) ;;
    *) echo "Invalid ALLOW_STATS_KEY_MISMATCH=${ALLOW_STATS_KEY_MISMATCH}"; exit 1 ;;
  esac
fi

if [[ -n "${ALLOW_ACTION_DIM_MISMATCH:-}" ]]; then
  case "${ALLOW_ACTION_DIM_MISMATCH,,}" in
    true|1|yes|y|on) ARGS+=(--allow_action_dim_mismatch) ;;
    false|0|no|n|off) ;;
    *) echo "Invalid ALLOW_ACTION_DIM_MISMATCH=${ALLOW_ACTION_DIM_MISMATCH}"; exit 1 ;;
  esac
fi

"${ARGS[@]}"
