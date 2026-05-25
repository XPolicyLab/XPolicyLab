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
config_path="${LINGBOT_VLA_CONFIG_PATH:-configs/vla/robodojo_sim_arx_x5.yaml}"
data_path="${LINGBOT_VLA_DATA_PATH:-${data_setting}}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "[LingBot_VLA] config=${config_path}"
echo "[LingBot_VLA] data_path=${data_path}"
echo "[LingBot_VLA] checkpoint_dir=${ckpt_dir}"

cd "${POLICY_DIR}/lingbot_vla"
bash train_origin.sh tasks/vla/train_lingbotvla.py \
  "${config_path}" \
  --data.train_path "${data_path}" \
  --train.output_dir "${ckpt_dir}" \
  --train.seed "${seed}"