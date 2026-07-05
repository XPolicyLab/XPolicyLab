#!/bin/bash
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
#            [expert_data_num] [raw_task_dirs] [fps] [output_dir]
# expert_data_num: optional; empty = use all episodes.
# raw_task_dirs:   raw HDF5 task dir(s) under data/<bench_name>/; comma-separated
#                  to merge. Defaults to ${ckpt_name}.
set -e

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
raw_task_dirs=${6:-${ckpt_name}}
fps=${7:-30}
_default_output_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
output_dir=${8:-${_default_output_dir}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

args=(
    "${bench_name}"
    "${ckpt_name}"
    "${env_cfg_type}"
    "${action_type}"
    --raw-task-dirs "${raw_task_dirs}"
    --fps "${fps}"
    --output_dir "${output_dir}"
    --project-root "${ROOT_DIR}"
)
if [[ -n "${expert_data_num}" ]]; then
    args+=(--expert-data-num "${expert_data_num}")
fi

echo -e "\033[33m[A1 process_data] Converting HDF5 to LeRobot format...\033[0m"
python "${SCRIPT_DIR}/process_data.py" "${args[@]}"
echo -e "\033[33m[A1 process_data] Done.\033[0m"
