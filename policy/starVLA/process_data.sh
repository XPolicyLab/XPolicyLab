#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>"
    echo "Example: bash process_data.sh RoboDojo stack_bowls arx_x5 3500 joint"
    exit 1
fi

dataset_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${SCRIPT_DIR}/source_starvla/examples/XPolicyLab/train_files/convert_xpolicy_to_lerobot3.py" \
    --root_dir "${ROOT_DIR}" \
    --dataset_name "${dataset_name}" \
    --ckpt_name "${ckpt_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --expert_data_num "${expert_data_num}" \
    --action_type "${action_type}" \
    --output_dir "${SCRIPT_DIR}/data/${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
