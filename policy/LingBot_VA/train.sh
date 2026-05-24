#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NGPU
NGPU="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l | xargs)"
export CONFIG_NAME="${LINGBOT_VA_CONFIG_NAME:-robotwin30_train}"
export LINGBOT_VA_DATASET_PATH="${LINGBOT_VA_DATASET_PATH:-${POLICY_DIR}/data/${data_setting}}"

echo "[LingBot_VA] config=${CONFIG_NAME}"
echo "[LingBot_VA] dataset=${LINGBOT_VA_DATASET_PATH}"
echo "[LingBot_VA] checkpoint_dir=${ckpt_dir}"

bash "${POLICY_DIR}/lingbot_va/script/run_va_posttrain.sh" \
  --save-root "${ckpt_dir}" \
  --seed "${seed}"
