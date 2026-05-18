#!/bin/bash
set -e

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
fps=${6:-30}
_default_output_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
output_dir=${7:-${_default_output_dir}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "\033[33m[GO1 process_data] Converting HDF5 to LeRobot format...\033[0m"
python "${SCRIPT_DIR}/process_data.py" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    --fps "${fps}" \
    --instruction "Do your job." \
    --output_dir "${output_dir}"
echo -e "\033[33m[GO1 process_data] Done.\033[0m"