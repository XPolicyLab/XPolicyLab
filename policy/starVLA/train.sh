#!/bin/bash
set -e

dataset_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"

processed_name="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
data_root_dir="${SCRIPT_DIR}/data/${processed_name}"
run_id="${processed_name}-${seed}"
run_root_dir="${SCRIPT_DIR}/checkpoints"
config_yaml="${STARVLA_ROOT}/examples/Robotwin/train_files/starvla_train_arx.yaml"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

if [[ ! -d "${data_root_dir}/arx_x5" ]]; then
    bash "${SCRIPT_DIR}/process_data.sh" "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}"
fi

echo "[starVLA] data_root_dir=${data_root_dir}"
echo "[starVLA] run_id=${run_id}"
echo "[starVLA] action_dim=${action_dim}"

cd "${STARVLA_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch \
    --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --seed "${seed}" \
    --run_id "${run_id}" \
    --run_root_dir "${run_root_dir}" \
    --datasets.vla_data.data_root_dir "${data_root_dir}" \
    --datasets.vla_data.data_mix arx_x5 \
    --datasets.vla_data.lerobot_version v3.0 \
    --framework.action_model.action_dim "${action_dim}" \
    --framework.action_model.state_dim "${action_dim}"
