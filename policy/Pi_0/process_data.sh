#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${OPENPI_DATA_MODE:-image}"

py_args=(
  "${bench_name}"
  "${ckpt_name}"
  "${env_cfg_type}"
  "${action_type}"
  --mode "${mode}"
)
if [[ -n "${expert_data_num}" ]]; then
  py_args+=("${expert_data_num}")
fi
if [[ -n "${raw_task_dirs}" ]]; then
  py_args+=(--raw-task-dirs "${raw_task_dirs}")
fi

cd "${POLICY_DIR}/openpi"
python scripts/process_data.py "${py_args[@]}"
