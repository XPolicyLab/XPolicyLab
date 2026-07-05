#!/bin/bash
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> [raw_task_dirs] [dataset_id]
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
expert_data_num=${4:?expert_data_num required}
action_type=${5:?action_type required}
raw_task_dirs=${6:-${ckpt_name}}
dataset_id=${7:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ADAPTER_DIR="${SCRIPT_DIR}/LDA-1B/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"
out_tag="$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
echo "[process_data] ${bench_name}/${raw_task_dirs}/${env_cfg_type} x${expert_data_num} (${action_type}) -> data/${out_tag}/"

cmd=(python "${ADAPTER_DIR}/process_data.py"
  --root-dir "${ROOT_DIR}"
  --policy-dir "${SCRIPT_DIR}"
  --bench-name "${bench_name}"
  --ckpt-name "${ckpt_name}"
  --raw-task-dirs "${raw_task_dirs}"
  --env-cfg-type "${env_cfg_type}"
  --expert-data-num "${expert_data_num}"
  --action-type "${action_type}")
if [[ -n "${dataset_id}" ]]; then
  cmd+=(--dataset-id "${dataset_id}")
fi
"${cmd[@]}"
