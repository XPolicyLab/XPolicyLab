#!/usr/bin/env bash
set -euo pipefail

# RoboTwin2 generalist post-training (multi-task LeRobot mixture).
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-none}"
export PRETRAINED_MODEL_FORMAT=native
export LEARNING_RATE="${LEARNING_RATE:-6e-5}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-30000}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_ID:-}" ]]; then
    export NUM_GPUS=8
fi
export QWEN_VARIANT="${QWEN_VARIANT:-qwen3-4b}"
export WSP_AUTO_DOWNLOAD="${WSP_AUTO_DOWNLOAD:-false}"
export WSP_VALIDATE_CHECKPOINT_ARTIFACTS="${WSP_VALIDATE_CHECKPOINT_ARTIFACTS:-false}"

export WSP_TASK="${WSP_TASK:-robotwin2}"
export RUN_NAME="${RUN_NAME:-stage1_posttrain_clean_robotwinconcat}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/robotwin2/auto-none/$RUN_NAME}"
export WSP_MODE=auto
export VISUAL_PROMPT=none
export NATIVE_DATASET_NAME=worldscape_lerobot_text
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT=wsp2
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"

# Keep credentials outside the tracked launcher. The file should contain:
#   WANDB_API_KEY=<your key>
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$HOME/.config/wandb/worldscape.env}"
if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$WANDB_ENV_FILE"
fi
if [[ "$WANDB_ENABLED" == "true" && "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARNING: WANDB_API_KEY is unset; relying on existing wandb login credentials" >&2
fi

export ACTION_MODE=joint
export RELATIVE_ACTION=false
export ACTION_HORIZON="${ACTION_HORIZON:-24}"
case "$ACTION_HORIZON" in
    24|48) ;;
    *)
        echo "ERROR: ACTION_HORIZON must be 24 or 48" >&2
        exit 2
        ;;
esac
: "${DATA_ROOT:?Set DATA_ROOT to the RoboTwin LeRobot dataset root}"
: "${ZSCORE_STATS_PATH:?Set ZSCORE_STATS_PATH to the RoboTwin dataset_stats.json}"
export DATA_ROOT ZSCORE_STATS_PATH
export VLM_HISTORY_NUM_FRAMES="${VLM_HISTORY_NUM_FRAMES:-4}"
export VLM_HISTORY_STRIDE="${VLM_HISTORY_STRIDE:-$ACTION_HORIZON}"
export VLM_HISTORY_WINDOW="${VLM_HISTORY_WINDOW:-$((4 * ACTION_HORIZON))}"
export WAM_MAX_CHUNK_SIZE="${WAM_MAX_CHUNK_SIZE:-1}"
# Temporal packing emits 8 video frames per chunk plus one endpoint.
export WAM_NUM_FRAMES="${WAM_NUM_FRAMES:-$((8 * WAM_MAX_CHUNK_SIZE + 1))}"
export WSP_DIFFUSION_VIEW_LAYOUT="${WSP_DIFFUSION_VIEW_LAYOUT:-mosaic_2x2}"

case "$WSP_DIFFUSION_VIEW_LAYOUT" in
    mosaic_2x2) ;;
    robotwin_concat)
        # Preserve native camera pixels for diffusion. `video` still receives
        # the normal crop/resize once for Qwen, while `video_raw` is composed
        # directly and each view is resized once into the 384x320 layout.
        export CACHE_VIDEO_RESIZE_WIDTH=null
        export CACHE_VIDEO_RESIZE_HEIGHT=null
        export RAW_VIDEO_OBSERVATION_KEY=video_raw
        export DIFFUSION_IMAGE_KEY=video_raw
        export WAM_FRAME_SEQLEN=120
        ;;
    *)
        echo "ERROR: WSP_DIFFUSION_VIEW_LAYOUT must be mosaic_2x2 or robotwin_concat" >&2
        exit 2
        ;;
esac

# shellcheck source=recipes/common/launcher_common.sh
source "$ROOT/recipes/common/launcher_common.sh"
wsp_distributed_train_launch "$ROOT" "$ROOT/configs/posttrain/robotwin2.yaml" "$@"
