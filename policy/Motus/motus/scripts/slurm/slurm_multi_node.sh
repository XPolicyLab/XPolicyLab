#!/bin/bash
# SLURM script for multi-node distributed training

#SBATCH --job-name=motus_multi
#SBATCH --output=/path/to/Motus/logs/slurm_multi_%j.out
#SBATCH --error=/path/to/Motus/logs/slurm_multi_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=256
#SBATCH --mem=1500G
#SBATCH --partition=xxx  # change here
#SBATCH --exclusive

echo "Starting multi-node job on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"

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

# Get node information
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
master_addr=$(echo "$nodes" | head -n 1)
export MASTER_ADDR=$master_addr

echo "NODELIST: $nodes"
echo "MASTER_ADDR: $master_addr"
echo "Current node index: $SLURM_NODEID"

# NCCL settings for multi-node
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

# Create logs directory
mkdir -p logs

# Training Configuration - Define here to avoid duplication in worker
CONFIG_FILE=${CONFIG_FILE:-"configs/robotwin.yaml"}
RUN_NAME=${RUN_NAME:-"robotwin_test"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=========================================="
echo "Multi-Node Training Configuration"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "Master addr: $master_addr"
echo "Master port: $MASTER_PORT"
echo "Config: $CONFIG_FILE"
echo "Run name: $RUN_NAME"
echo "=========================================="

# Export configuration variables for worker script
export CONFIG_FILE
export RUN_NAME
export MASTER_PORT

# Multi-node distributed training - use srun to launch worker on all nodes
srun bash scripts/slurm/multi_node_worker.sh

echo "Training completed at $(date)"