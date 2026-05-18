#!/usr/bin/bash
# ------------------ Get Number of GPUs ------------------ #
n_gpus=$(( $(echo "$SLURM_JOB_GPUS" | tr -cd ',' | wc -c) + 1 )) # Get GPU count from environment variable
# ------------------- Setup Environment ------------------ #
export MASTER_ADDR=$SLURMD_NODENAME          # Get master node address
export MASTER_PORT=$((RANDOM % 1010 + 20000)) # Randomly generate communication port to avoid conflicts; uses random number between 20000-20100, can be modified as needed
export NNODES=$SLURM_JOB_NUM_NODES           # Get number of nodes
export NPROC_PER_NODE=${n_gpus}              # Get number of GPUs per node
# ------------------------- Main ------------------------- #
srun bash scripts/slurms/pretrain.sh # Use `srun` to distribute training command to all nodes, i.e., each node will execute `bash train.sh`
