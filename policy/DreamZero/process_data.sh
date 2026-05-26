#!/bin/bash
set -e
set -o pipefail

dataset_name=${1:?dataset_name is required}
task_name=${2:?task_name is required}
env_cfg_type=${3:?env_cfg_type is required}
expert_data_num=${4:?expert_data_num is required}
action_type=${5:?action_type is required}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fps="${DREAMZERO_FPS:-30}"
output_dir="${DREAMZERO_DATA_DIR:-${SCRIPT_DIR}/data}"

python "${SCRIPT_DIR}/process_data.py" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    --fps "${fps}" \
    --output_dir "${output_dir}"