#!/bin/bash

task_name=${1}
env_cfg_type=${2}
expert_data_num=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}

# Get Action Dimension from env_cfg_type
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"

export ACT_ACTION_DIM=${action_dim}

python3 imitate_episodes.py \
    --task_name ${task_name}-${env_cfg_type}-${expert_data_num}-${action_type} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${env_cfg_type}-${expert_data_num}-${action_type} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 6000 \
    --seed ${seed}
