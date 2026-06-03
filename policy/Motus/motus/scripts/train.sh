#!/bin/bash
# Define your env settings here 
# e.g., nccl, network, proxy, etc.

TASK="robotwin"  # Define your task name here
CONFIG_FILE="configs/robotwin.yaml"  # Define your dataset config path here

export OUTPUT_DIR="outputs/motus-${TASK}" # Define your output directory here

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Single-node training with torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config $CONFIG_FILE \
    --run_name $TASK \
    --report_to tensorboard
