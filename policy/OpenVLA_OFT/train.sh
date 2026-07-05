#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
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