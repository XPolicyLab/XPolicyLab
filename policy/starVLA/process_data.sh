#!/bin/bash
set -e

dataset_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/starvla_adapter/data/convert_xpolicy_to_lerobot3.py" \
    --root_dir "${ROOT_DIR}" \
    --dataset_name "${dataset_name}" \
    --ckpt_name "${ckpt_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --expert_data_num "${expert_data_num}" \
    --action_type "${action_type}" \
    --output_dir "${SCRIPT_DIR}/data/${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
