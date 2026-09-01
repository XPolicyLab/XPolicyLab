#!/bin/bash
#SBATCH --job-name=ola_sem_train_smoke
#SBATCH --partition=acd_u
#SBATCH --output=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM/logs/train_smoke_%j.out
#SBATCH --error=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM/logs/train_smoke_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM

set -euo pipefail
source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/wsong890/envs/motus
module load cuda/12.8
module load ffmpeg/6.0.1

export OLA_SEM_DATASET_ROOT=/data/user/wsong890/user68/cjy/Motus/data/robotwin_dataset
export OLA_SEM_WAN_PATH=/data/user/wsong890/user68/cjy/Motus/pretrained_models/Wan2.2-TI2V-5B
export OLA_SEM_VLM_PATH=/data/user/wsong890/user68/cjy/Motus/pretrained_models/Qwen3-VL-2B-Instruct
export OLA_SEM_FINETUNE_CHECKPOINT=${OLA_SEM_FINETUNE_CHECKPOINT:-/data/user/wsong890/user68/cjy/Motus/pretrained_models/d0_v}
export OLA_SEM_OUTPUT_ROOT=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM/checkpoints-smoke
export OLA_SEM_MAX_STEPS=1
export OLA_SEM_MAX_EPISODES=16
export OLA_SEM_BATCH_SIZE=1
export OLA_SEM_NUM_WORKERS=0
export OLA_SEM_REPORT_TO=none
run_id=${SLURM_JOB_ID:-$$}
export MASTER_PORT=$((20000 + run_id % 20000))
export TRITON_CACHE_DIR=/tmp/ola_sem_triton_cache_${run_id}
mkdir -p "${TRITON_CACHE_DIR}"

bash process_data.sh RoboTwin smoke aloha_agilex joint "${OLA_SEM_DATASET_ROOT}"
bash train.sh RoboTwin smoke aloha_agilex joint 42 0,1,2,3,4,5,6,7 8
