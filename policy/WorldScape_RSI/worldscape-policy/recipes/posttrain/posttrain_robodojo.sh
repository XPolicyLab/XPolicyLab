#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
# The RoboDojo loader uses XPolicyLab's shared image decoder. Make the
# containing checkout importable even from the policy/runtime working directory.
XPL_ROOT="$(cd -- "$ROOT/../../.." && pwd -P)"
if [[ ! -f "$XPL_ROOT/utils/process_data.py" || ! -f "$XPL_ROOT/__init__.py" ]]; then
    echo "ERROR: RoboDojo training requires the bundled XPolicyLab checkout: $XPL_ROOT" >&2
    exit 1
fi
export PYTHONPATH="$(dirname -- "$XPL_ROOT")${PYTHONPATH:+:$PYTHONPATH}"

# Public RoboDojo z-score recipe: flash (H24/C1) or memory (H24/C24).
source "$ROOT/recipes/common/robodojo_model.sh"
config="$ROOT/configs/posttrain/$robodojo_config_name"
export DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the RoboDojo dataset root}"
export ROBODOJO_ZSCORE_STATS="${ROBODOJO_ZSCORE_STATS:?Set ROBODOJO_ZSCORE_STATS to the z-score JSON artifact}"
export ROBODOJO_INDEX_PATH="${ROBODOJO_INDEX_PATH:?Set ROBODOJO_INDEX_PATH to the index matching the dataset, horizon, chunks and stride}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ -z "${NUM_GPUS:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    export NUM_GPUS="${#visible_gpus[@]}"
fi
export RUN_NAME="${RUN_NAME:-baseline}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/../checkpoints/$RUN_NAME}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/wandb}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:?Set PRETRAINED_MODEL_PATH to a native pretrained checkpoint}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/tmp}"
export WSP_CACHE_DIR="${WSP_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/worldscape-policy}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$WSP_CACHE_DIR/cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WSP_CACHE_DIR/matplotlib}"
export HF_HOME="${HF_HOME:-$WSP_CACHE_DIR/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WSP_CACHE_DIR/xdg}"
# No dependency on the source checkout's local reports or timing instrumentation.
export WSP_TIMING_TRACE_DIR="${WSP_TIMING_TRACE_DIR:-}"

export CONDA_ENV="${CONDA_ENV:-${CONDA_PREFIX:-}}"
export WSP_MODEL_ROOT="${WSP_MODEL_ROOT:-$ROOT/../checkpoints}"
export WAN_CKPT_DIR="${WAN_CKPT_DIR:-$WSP_MODEL_ROOT/Wan2.2-TI2V-5B}"
export CLIP_CKPT_DIR="${CLIP_CKPT_DIR:-$WSP_MODEL_ROOT/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$WSP_MODEL_ROOT/umt5-xxl}"
export Qwen_CKPT_DIR="${Qwen_CKPT_DIR:-$WSP_MODEL_ROOT/Qwen3-VL-4B-Instruct}"
export DEEPSPEED_MODE="${DEEPSPEED_MODE:-zero2}"
export QWEN_VARIANT="${QWEN_VARIANT:-qwen3-4b}"
export WSP_AUTO_DOWNLOAD="${WSP_AUTO_DOWNLOAD:-false}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: RoboDojo dataset not found at $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$ROBODOJO_INDEX_PATH" ]]; then
    echo "ERROR: existing RoboDojo index not found: $ROBODOJO_INDEX_PATH" >&2
    exit 1
fi
if [[ ! -f "$PRETRAINED_MODEL_PATH/checkpoint_manifest.json" ]]; then
    echo "ERROR: RoboDojo training requires a native checkpoint: $PRETRAINED_MODEL_PATH" >&2
    exit 1
fi

# Shared checks do not launch another recipe. Final CLI overrides take precedence.
source "$ROOT/recipes/common/robodojo_zscore.sh"
source "$ROOT/recipes/common/launcher_common.sh"
wsp_distributed_train_launch "$ROOT" "$config" "$@"
