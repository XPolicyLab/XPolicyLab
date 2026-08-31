#!/bin/bash
#SBATCH --job-name=ola_sem_debug
#SBATCH --partition=acd_u
#SBATCH --output=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM/logs/debug_%j.out
#SBATCH --error=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM/logs/debug_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/user/wsong890/user68/cjy/Motus/XPolicyLab/policy/OLA_SEM

set -euo pipefail
source /share/anaconda3/etc/profile.d/conda.sh
module load cuda/12.8

export EVAL_ENV_TYPE=debug
export DEBUG_OBS_ENCODED=${DEBUG_OBS_ENCODED:-0}
export OLA_SEM_WAN_PATH=/data/user/wsong890/user68/cjy/Motus/pretrained_models/Wan2.2-TI2V-5B
export OLA_SEM_VLM_PATH=/data/user/wsong890/user68/cjy/Motus/pretrained_models/Qwen3-VL-2B-Instruct

checkpoint=/data/user/wsong890/user68/cjy/Motus/checkpoints/robotwin_lap_history_flow_clean/robotwin_clean_lap_history_flow_clean/checkpoint_step_30000
bash eval.sh RoboTwin hanging_mug "${checkpoint}" aloha_agilex joint 42 0 0 \
    /data/user/wsong890/envs/motus \
    /data/user/wsong890/user68/conda_env/robotwin_motus
