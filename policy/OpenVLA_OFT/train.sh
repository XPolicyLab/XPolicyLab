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
data_setting="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
tfds_dataset_name="${OPENVLA_TFDS_DATASET_NAME:-aloha_${data_setting}}"

mkdir -p "${ckpt_dir}"

echo "[OpenVLA_OFT] tfds_dataset_name=${tfds_dataset_name}"
echo "[OpenVLA_OFT] checkpoint_dir=${ckpt_dir}"

bash "${POLICY_DIR}/openvla_oft/scripts/finetune.sh" \
  "${ckpt_dir}" \
  "${tfds_dataset_name}" \
  "${gpu_id}" \
  "${seed}"