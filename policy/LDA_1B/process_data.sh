#!/bin/bash
set -euo pipefail

dataset_name=${1}
task_name=${2}          # single task, or comma-separated list to merge, e.g. "test_data,test_data_1"
env_cfg_type=${3}
expert_data_num=${4}    # episodes kept PER task
action_type=${5}
dataset_id=${6:-}       # optional output folder name; default cotrain_dataset for multi-task

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cmd=(python "${ROOT_DIR}/XPolicyLab/policy/LDA_1B/LDA-1B/xpolicylab_adapter/process_data.py"
  --root-dir "${ROOT_DIR}"
  --dataset-name "${dataset_name}"
  --task-name "${task_name}"
  --env-cfg-type "${env_cfg_type}"
  --expert-data-num "${expert_data_num}"
  --action-type "${action_type}")
if [[ -n "${dataset_id}" ]]; then
  cmd+=(--dataset-id "${dataset_id}")
fi
"${cmd[@]}"
