#!/bin/bash
set -e
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bash process_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type>

Optional environment overrides:
  DREAMZERO_DATA_DIR        Default: <policy>/data
  DREAMZERO_FPS             Default: 25
EOF
}

if [ "$#" -ne 5 ]; then
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fps="${DREAMZERO_FPS:-25}"
output_dir="${DREAMZERO_DATA_DIR:-${SCRIPT_DIR}/data}"

dataset_name=$1
task_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5

python "${SCRIPT_DIR}/process_data.py" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    --source_format hdf5 \
    --fps "${fps}" \
    --output_dir "${output_dir}"