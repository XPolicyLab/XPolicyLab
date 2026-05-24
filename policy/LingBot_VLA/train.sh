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