#!/bin/bash

bench_name=${1}
ckpt_name=${2} # run name
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get Action Dimension from env_cfg_type
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"

export ACT_ACTION_DIM=${action_dim}

ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

python3 imitate_episodes.py \
    --bench_name ${bench_name} \
    --task_name ${ckpt_name} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir "${SCRIPT_DIR}/checkpoints/${ckpt_setting}-${seed}" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 16 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 6000 \
    --seed ${seed}
