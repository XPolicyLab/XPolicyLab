#!/usr/bin/env bash
set -euo pipefail

# Native Fold Shirt parity launcher.
# Data flow inherited from fold_shirt_training_wan22_freeze_vlm_lstm.sh:
#   - Auto checkpoint with one VLM/T5 prompt branch selected per batch
#   - VLM receives the high-level task; T5 receives only the atomic subtask
#   - text-only visual condition (VISUAL_PROMPT=none)
#   - HDF5 event data with 4-chunk temporal packing
#   - 33 packed video frames, 96 packed action steps, action horizon 24
#   - VLM history: 8 frames, stride 24, window 192
#   - frozen VLM/T5, final-layer perception + AR planning tokens by default
#   - optional trainable QFormer via VLM_TOKEN_MODE=qformer FREEZE_QFORMER=false
#   - trainable projectors/WAM/action adapters/event memory
#   - semantic forcing weight 0.001 on Auto-routed batches only; planning CE disabled
#   - Auto ratio 0.5 through 30% progress, then 0.7
#
# Fold-specific defaults:
#   PRETRAINED_MODEL_PATH  Native WSP checkpoint; defaults to none
#   OUTPUT_DIR             Experiment checkpoints; derived from task/mode/run by default
#   DATA_ROOT              Self-contained dataset root (episodes + merged meta/)
#   MAX_STEPS             Optimizer steps; defaults to 100000
#   LEARNING_RATE         Base learning rate; defaults to 6e-5
#   LR_SCHEDULE           Learning-rate schedule; defaults to cosine
#   PER_DEVICE_TRAIN_BATCH_SIZE  Per-GPU batch size; defaults to 8
#   NUM_GPUS              Processes/GPUs per node; defaults to 8
#   QWEN_VARIANT          qwen3-4b|qwen3-2b; defaults to qwen3-4b
# Common environment, model, distributed, and optimizer defaults are owned by
# recipes/common/posttrain_agilex.sh.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export MAX_STEPS="${MAX_STEPS:-100000}"
export LEARNING_RATE="${LEARNING_RATE:-6e-5}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_ID:-}" ]]; then
    export NUM_GPUS=8
fi
export QWEN_VARIANT="${QWEN_VARIANT:-qwen3-4b}"

# export DEEPSPEED_MODE=zero2_offload
# export MAX_STEPS=100
# export SAVE_STEPS=100
# export RUN_NAME=smoke-100steps
# export NUM_GPUS=1
# export PER_DEVICE_TRAIN_BATCH_SIZE=1
# export LOG_PROMPT_TEXT="${LOG_PROMPT_TEXT:-false}"

: "${DATA_ROOT:?Set DATA_ROOT to the self-contained Fold Shirt HDF5 dataset}"
export DATA_ROOT
export WSP_TASK="fold-shirt-text"
export RUN_NAME="${RUN_NAME:-baseline}"
export WSP_MODE="${WSP_MODE:-auto}"
export VISUAL_PROMPT="none"
export NATIVE_DATASET_NAME="worldscape_hdf5_text"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/${WSP_TASK}/${WSP_MODE}-${VISUAL_PROMPT}/${RUN_NAME}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-world-scape-policy}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [[ ! -f "$DATA_ROOT/meta/episodes.jsonl" ]]; then
    echo "ERROR: Expected merged dataset meta at $DATA_ROOT/meta/episodes.jsonl"
    exit 1
fi

exec "$ROOT/recipes/common/posttrain_agilex.sh" "$@"
