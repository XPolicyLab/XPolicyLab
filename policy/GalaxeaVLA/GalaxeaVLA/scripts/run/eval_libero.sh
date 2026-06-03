#!/bin/bash
# This script is used to eval model in LIBERO using multiple GPUs.
# arguments:
#   GPU number
#   task config
#   checkpoint path
#   other hydra overrides

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ENABLE_MONITORING=0 # Disable NCCL monitoring/heartbeat completely

GPU=$1
config=$2
ckpt_path=$3
ARGS=${@:4}

config="${config#configs/}" # delete prefix configs/
config="${config#task/}" # delete prefix task/
config="${config%.yaml}" # delete suffix .yaml

torchrun --standalone --nnodes 1 --nproc-per-node $GPU \
    scripts/eval_libero.py \
    task=$config \
    ckpt_path=$ckpt_path \
    logger.mode=local \
    $ARGS