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
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
converted_data_root="${SPIRIT_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
raw_data_root="${SPIRIT_RAW_DATA_ROOT:-${ROOT_DIR}/data}"
patterns_csv="${SPIRIT_PATTERNS_CSV:-${dataset_name}.${ckpt_name}.${env_cfg_type}}"
pretrained_path="${SPIRIT_PRETRAINED_PATH:-/mnt/xspark-data/xspark_shared/model_weights/Spirit-v1.5}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
num_gpus="$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "[Spirit_v15] raw_data_root=${raw_data_root}"
echo "[Spirit_v15] patterns_csv=${patterns_csv}"
echo "[Spirit_v15] converted_data_root=${converted_data_root}"
echo "[Spirit_v15] checkpoint_dir=${ckpt_dir}"

bash "${POLICY_DIR}/spirit_v15/scripts/train_xpolicylab_from_raw.sh" \
  "${raw_data_root}" \
  "${patterns_csv}" \
  "${converted_data_root}" \
  "${pretrained_path}" \
  "${ckpt_dir}" \
  "${SPIRIT_NUM_GPUS:-${num_gpus}}" \
  "${SPIRIT_BATCH_SIZE:-32}" \
  "${SPIRIT_MAX_TRAIN_STEPS:-40000}" \
  "${SPIRIT_LOG_INTERVAL:-25}" \
  "${SPIRIT_SAVE_STEPS:-2500}" \
  "${SPIRIT_NUM_WORKERS:-4}" \
  "${SPIRIT_PREFETCH_FACTOR:-8}" \
  "${SPIRIT_WANDB_MODE:-disabled}" \
  "${ckpt_name}" \
  "${SPIRIT_TASK_PROMPT:-Perform the instructed bimanual manipulation task.}" \
  "${SPIRIT_FPS:-auto}" \
  "${SPIRIT_OVERWRITE_DATASET:-0}" \
  "${SPIRIT_MAX_EPISODES_PER_TARGET:-${expert_data_num}}" \
  "${SPIRIT_ROBOT_TYPE:-aloha}" \
  "${dataset_name}" \
  "${SPIRIT_DATA_VERSION:-v1.0}" \
  "${SPIRIT_SKIP_CONVERT:-0}" \
  "${SPIRIT_CONVERT_ONLY:-0}"
