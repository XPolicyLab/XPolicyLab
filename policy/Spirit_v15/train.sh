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
converted_data_root="${SPIRIT_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
pretrained_path="${SPIRIT_PRETRAINED_PATH:-/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Spirit-v1.5}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

if [[ ! -f "${converted_data_root}/meta/task_info.json" ]]; then
  echo "Converted dataset not found: ${converted_data_root}/meta/task_info.json" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

mkdir -p "${ckpt_dir}"

echo "[Spirit_v15] converted_data_root=${converted_data_root}"
echo "[Spirit_v15] pretrained_path=${pretrained_path}"
echo "[Spirit_v15] checkpoint_dir=${ckpt_dir}"
echo "[Spirit_v15] seed=${seed}"
echo "[Spirit_v15] gpu_id=${gpu_id}"

bash "${POLICY_DIR}/spirit_v15/scripts/train_xpolicylab_converted.sh" \
  "${converted_data_root}" \
  "${pretrained_path}" \
  "${ckpt_dir}" \
  "${gpu_id}" \
  "${SPIRIT_BATCH_SIZE:-32}" \
  "${SPIRIT_MAX_TRAIN_STEPS:-50000}" \
  "${SPIRIT_LOG_INTERVAL:-25}" \
  "${SPIRIT_SAVE_STEPS:-2500}" \
  "${SPIRIT_NUM_WORKERS:-4}" \
  "${SPIRIT_PREFETCH_FACTOR:-8}" \
  "${SPIRIT_WANDB_MODE:-online}" \
  "${seed}"
