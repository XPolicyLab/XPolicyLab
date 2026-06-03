#!/bin/bash
# Multi-node worker script - runs on each node
# Environment setup is done in main slurm_multi_node.sh script
# This worker inherits configuration variables from parent
echo "Starting worker on $(hostname) at $(date)"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"

# Get master node address (inherited from parent environment)
if [ -z "$SLURM_JOB_ID" ]; then
    master_addr=127.0.0.1
else
    nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    master_addr=$(echo "$nodes" | head -n 1)
fi

# Configuration - all now inherited from main script
echo "Worker configuration:"
echo "  Node: $(hostname)"
echo "  Node rank: $SLURM_NODEID"
echo "  Master addr: $master_addr"
echo "  Master port: $MASTER_PORT"
echo "  Config: $CONFIG_FILE"
echo "  Run name: $RUN_NAME"

# Multi-node distributed training with torchrun + DeepSpeed
torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$master_addr \
    --master_port=$MASTER_PORT \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config $CONFIG_FILE \
    $(if [ -n "$RUN_NAME" ]; then echo "--run_name $RUN_NAME"; fi) \
    --report_to tensorboard

echo "Worker on $(hostname) completed at $(date)"