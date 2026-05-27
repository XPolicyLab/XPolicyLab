#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

export MLP_WORKER_NUM=${WORLD_SIZE:-1}
export MLP_WORKER_GPU=${RESOURCE_GPU:-1}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29500}
export PYTHONPATH=$(pwd):${PYTHONPATH}

set -e -x

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    tools/train.py \
    $@
