#!/usr/bin/bash

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
# export NCCL_ASYNC_ERROR_HANDLING=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET

script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

echo nproc_per_node=$nproc_per_node
echo nnodes=$nnodes
echo node_rank=$node_rank
echo master_addr=$master_addr
echo master_port=$master_port


NGPU=`nvidia-smi --list-gpus | wc -l`
torchrun --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --node_rank=$node_rank \
    --master-addr $master_addr \
    --master-port $master_port \
    $script_path \
    --config_file $config_path




