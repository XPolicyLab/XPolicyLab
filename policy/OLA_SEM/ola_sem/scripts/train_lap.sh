#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_FILE="${CONFIG_FILE:-configs/robotwin_lap.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
NUM_GPUS="${NUM_GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ola-sem-robotwin}"
RUN_NAME="${RUN_NAME:-robotwin_lap}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
REPORT_TO="${REPORT_TO:-tensorboard}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: training config not found: $CONFIG_FILE" >&2
    exit 1
fi
if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    echo "Error: DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2
    exit 1
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_GPUS must be a positive integer" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$REPO_ROOT/bak:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec torchrun \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config "$CONFIG_FILE" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO"
