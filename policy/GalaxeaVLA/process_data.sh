#!/bin/bash
# Convert XPolicyLab HDF5 episodes -> Galaxea LeRobot format for fine-tuning.
# Output: policy/GalaxeaVLA/data/<tag>-lerobot/ (point a task config's dataset_dirs there).
set -euo pipefail

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5:-joint}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

echo "[process_data] standardizing every camera frame to RGB HWC (240, 320, 3)"
echo "[process_data] ${dataset_name}/${task_name}/${env_cfg_type} x${expert_data_num} (${action_type})"

source "${UPSTREAM_DIR}/.venv/bin/activate"
PYTHONPATH="${ROOT_DIR}:${UPSTREAM_DIR}/src:${PYTHONPATH:-}" \
python "${UPSTREAM_DIR}/xpolicylab_adapter/convert_to_galaxea_lerobot.py" \
    "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}"
