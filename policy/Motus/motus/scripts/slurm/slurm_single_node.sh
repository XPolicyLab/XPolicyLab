#!/bin/bash
# SLURM script for single node 8-GPU training
# Usage: sbatch scripts/slurm_single_node.sh

#SBATCH --job-name=motus
#SBATCH --output=/path/to/Motus/logs/slurm_single_%j.out
#SBATCH --error=/path/to/Motus/logs_/slurm_single_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=256
#SBATCH --mem=1500G
#SBATCH --partition=xxx  # change here
#SBATCH --exclusive

echo "Starting single node job on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Setup environment
PROJECT_ROOT="/path/to/Motus"
cd $PROJECT_ROOT

# Load modules and activate conda environment
module load cuda/12.8 || echo "Warning: Could not load CUDA module"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/motus_env

# Set environment variables
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export OMP_NUM_THREADS=8
export CUDA_HOME=$CONDA_PREFIX

# Get master node address (single node -> localhost is fine)
if [ -z "$SLURM_JOB_ID" ]; then
    master_addr=127.0.0.1
else
    nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    master_addr=$(echo "$nodes" | head -n 1)
fi

# NCCL settings for better performance
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_13:1,mlx5_16:1,mlx5_17:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=23
export NCCL_DEBUG=INFO

# Increase timeout for checkpoint saving (default is 600s/10min, set to 30min)
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# Create logs directory (for any additional logs)
mkdir -p /path/to/Motus/logs

CONFIG_FILE=${CONFIG_FILE:-"configs/robotwin.yaml"}
RUN_NAME=${RUN_NAME:-"robotwin_test"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Worker configuration:"
echo "  Node: $(hostname)"
echo "  Node rank: $SLURM_NODEID"
echo "  Master addr: $master_addr"
echo "  Master port: $MASTER_PORT"
echo "  Config: $CONFIG_FILE"
echo "  Run name: $RUN_NAME"
echo "  Resume From (YAML): resume.checkpoint_path"
echo "  Finetune From (YAML): finetune.checkpoint_path"

# Single node training with torchrun + DeepSpeed
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
    --report_to tensorboard \

echo "Training completed at $(date)"