#!/bin/bash
set -e

dataset_name=$1
task_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
gpu_id=$6
seed=$7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_SCRIPT="${SCRIPT_DIR}/setup_train_data.sh"
POLICY_SCRIPT="${SCRIPT_DIR}/setup_train_policy.sh"

bash "${DATA_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}"

bash "${POLICY_SCRIPT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    "${gpu_id}" \
    "${seed}"
