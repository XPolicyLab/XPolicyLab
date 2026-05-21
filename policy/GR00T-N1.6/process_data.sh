#!/bin/bash
set -e
set -o pipefail

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
fps=${6:-30}
output_dir=${7:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${SCRIPT_DIR}/Isaac-GR00T:${PYTHONPATH}"

if [ -z "${output_dir}" ]; then
    output_dir="${SCRIPT_DIR}/data"
fi

python "${SCRIPT_DIR}/process_data.py" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    --fps "${fps}" \
    --output-dir "${output_dir}"
