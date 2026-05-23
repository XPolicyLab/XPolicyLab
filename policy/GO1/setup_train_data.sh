#!/bin/bash
set -e

dataset_name=$1
task_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

repo_id="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
lerobot_output_dir="${SCRIPT_DIR}/data"
lerobot_data_path="${lerobot_output_dir}/${repo_id}"

echo -e "\033[33m[INFO] Checking if LeRobot dataset exists at: ${lerobot_data_path}\033[0m"

if [ -d "${lerobot_data_path}" ]; then
    echo -e "\033[33m[INFO] LeRobot dataset '${repo_id}' already exists, skipping conversion.\033[0m"
else
    echo -e "\033[33m[INFO] Converting HDF5 data to LeRobot format...\033[0m"
    bash "${SCRIPT_DIR}/process_data.sh" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" 30 "${lerobot_output_dir}"
fi

echo -e "\033[33m[INFO] LeRobot data path: ${lerobot_data_path}\033[0m"

