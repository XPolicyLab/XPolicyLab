#!/bin/bash
#SBATCH --job-name=ola_sem_robotwin_eval
#SBATCH --partition=acd_u
#SBATCH --output=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM/logs/eval_robotwin_%j.out
#SBATCH --error=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM/logs/eval_robotwin_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM

set -euo pipefail
source /share/anaconda3/etc/profile.d/conda.sh
module load cuda/12.8
module load ffmpeg/6.0.1

export EVAL_ENV_TYPE=sim
export OLA_SEM_WAN_PATH=/data/user/wsong890/user68/cjy/Motus/pretrained_models/Wan2.2-TI2V-5B
export OLA_SEM_VLM_PATH=/data/user/wsong890/user68/cjy/Motus/pretrained_models/Qwen3-VL-2B-Instruct
export OLA_SEM_ROBOTWIN_ROOT=/data/user/wsong890/user68/cjy/Motus/RoboTwin
export OLA_SEM_ROBOTWIN_TASK_CONFIG=demo_clean
export OLA_SEM_TEST_NUM=1

checkpoint=/data/user/wsong890/user68/cjy/Motus/checkpoints/robotwin_lap_history_flow_clean/robotwin_clean_lap_history_flow_clean/checkpoint_step_30000
bash eval.sh RoboTwin hanging_mug "${checkpoint}" aloha_agilex joint 42 0 0 \
    /data/user/wsong890/envs/motus \
    /data/user/wsong890/user68/conda_env/robotwin_motus
