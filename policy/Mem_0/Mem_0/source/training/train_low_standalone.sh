#!/usr/bin/env bash
set -euo pipefail

# Activate conda env and launch 8-GPU training.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mem0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29500}
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TORCH_CPP_LOG_LEVEL=ERROR

if [ "$#" -eq 0 ]; then
  ARGS=(--config "${CONFIG:-source/config/execution_module_train.yaml}")
else
  ARGS=("$@")
fi

torchrun \
  --standalone \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --nnodes=1 \
  --nproc_per_node=8 \
  source/training/train_low.py \
  "${ARGS[@]}"
