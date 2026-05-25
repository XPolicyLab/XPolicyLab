#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES_VALUE"
DEFAULT_NPROC_PER_NODE="${#CUDA_DEVICE_ARRAY[@]}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$DEFAULT_NPROC_PER_NODE}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2_stage2.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
RUN_NAME="${RUN_NAME:-sim_stack_bowls_v21_motus}"
REPORT_TO="${REPORT_TO:-tensorboard}"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config configs/lerobot_sim_stack_bowls.yaml \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO"