#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: $0 <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
gpu_id=$8

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
meta_path="${XVLA_META_PATH:-${POLICY_DIR}/xvla/meta.json}"
model_path="${XVLA_MODEL_PATH:-/mnt/xspark-data/xspark_shared/model_weights/X-VLA-Pt}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "[X_VLA] meta_path=${meta_path}"
echo "[X_VLA] checkpoint_dir=${ckpt_dir}"

accelerate launch \
    --mixed_precision bf16 \
    xvla/train.py \
    --models "${model_path}" \
    --train_metas_path "${meta_path}" \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 30000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --batch_size 32 \
    --output_dir "${ckpt_dir}" \
    --seed "${seed}" \
    --save_interval 1000