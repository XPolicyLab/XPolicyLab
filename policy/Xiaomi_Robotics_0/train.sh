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
XR0_ROOT="${POLICY_DIR}/xiaomi_robotics_0/xr0"
data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
converted_data_root="${XR0_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
data_config_name="${XR0_DATA_CONFIG_NAME:-${data_setting}}"
data_config_path="${XR0_ROOT}/configs/data/${data_config_name}.yaml"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
pretrained_path="${XR0_PRETRAINED_PATH:-${XR0_ROOT}/pretrained_ckpt/xr0_pretrained.pt}"

if [[ ! -d "${converted_data_root}/json" ]]; then
  echo "Converted dataset not found: ${converted_data_root}/json" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${data_config_path}" ]]; then
  echo "Data config not found: ${data_config_path}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${pretrained_path}" ]]; then
  echo "Pretrained checkpoint not found: ${pretrained_path}" >&2
  echo "Download Xiaomi-Robotics-0-Pretrain and run weight_convert.py, or set XR0_PRETRAINED_PATH." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export RESOURCE_GPU="${RESOURCE_GPU:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${ckpt_dir}"

echo "[Xiaomi_Robotics_0] converted_data_root=${converted_data_root}"
echo "[Xiaomi_Robotics_0] data_config=${data_config_path}"
echo "[Xiaomi_Robotics_0] pretrained_path=${pretrained_path}"
echo "[Xiaomi_Robotics_0] checkpoint_dir=${ckpt_dir}"
echo "[Xiaomi_Robotics_0] seed=${seed}"
echo "[Xiaomi_Robotics_0] gpu_id=${gpu_id}"
echo "[Xiaomi_Robotics_0] resource_gpu=${RESOURCE_GPU}"

cd "${XR0_ROOT}"

bash scripts/train.sh \
  "data=${data_config_name}" \
  "trainer.project=xr0" \
  "trainer.exp_name=${ckpt_setting}" \
  "trainer.default_root_dir=${ckpt_dir}" \
  "trainer.seed=${seed}" \
  "model.params.model.pretrained=${pretrained_path}" \
  "model.params.model.async_train=${XR0_ASYNC_TRAIN:-false}" \
  "trainer.max_steps=${XR0_MAX_STEPS:-30000}" \
  "trainer.save_interval=${XR0_SAVE_INTERVAL:-5000}"
