#!/bin/bash
set -e

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"
python process_data.py \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}"