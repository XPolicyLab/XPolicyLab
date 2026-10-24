#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the Build Block dataset}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [[ ! -f "$DATA_ROOT/meta/episodes.jsonl" ]]; then
    echo "ERROR: Expected merged dataset meta at $DATA_ROOT/meta/episodes.jsonl"
    exit 1
fi

export LEARNING_RATE="${LEARNING_RATE:-6e-5}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${GPU_ID:-}" ]]; then
    export NUM_GPUS=8
fi
export QWEN_VARIANT="${QWEN_VARIANT:-qwen3-4b}"
export WSP_TASK="build-block-demo"
export WSP_MODE="${WSP_MODE:-interactive}"
export VISUAL_PROMPT="demo"
export NATIVE_DATASET_NAME="worldscape_hdf5_demo"

exec "$ROOT/recipes/common/posttrain_agilex.sh" "$@"
