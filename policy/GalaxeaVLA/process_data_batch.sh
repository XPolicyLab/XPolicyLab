#!/bin/bash
# Batch convert: merge ALL tasks under <batch_root>/<task>/<env_cfg_type> into ONE
# multi-task Galaxea LeRobot dataset (per-episode instruction = task dir name).
# Output: policy/GalaxeaVLA/data/<dataset_name>-<env_cfg_type>-<action_type>-lerobot/
#
# Usage:
#   bash process_data_batch.sh <dataset_name> <env_cfg_type> <action_type> <batch_root> [max_episodes_per_task] [tasks...]
# Example (all tasks, all episodes):
#   bash process_data_batch.sh RoboDojo_first100 arx_x5 joint \
#       /mnt/xspark-data/zijian/final_data/RoboDojo_first100
set -euo pipefail

dataset_name=${1:?dataset_name required}
env_cfg_type=${2:?env_cfg_type required}
action_type=${3:?action_type required}
batch_root=${4:?batch_root required}
max_per_task=${5:-0}
shift $(( $# < 5 ? $# : 5 )) || true
tasks=("$@")   # optional subset of task dir names

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

echo "[process_data_batch] root=${batch_root} env=${env_cfg_type} action=${action_type} max/task=${max_per_task}"
echo "[process_data_batch] standardizing every camera frame to RGB HWC (240, 320, 3)"

tasks_arg=()
if [[ ${#tasks[@]} -gt 0 ]]; then
    tasks_arg=(--tasks "${tasks[@]}")
fi

source "${UPSTREAM_DIR}/.venv/bin/activate"
PYTHONPATH="${ROOT_DIR}:${UPSTREAM_DIR}/src:${PYTHONPATH:-}" \
python "${UPSTREAM_DIR}/xpolicylab_adapter/convert_to_galaxea_lerobot.py" \
    "${dataset_name}" "all" "${env_cfg_type}" "${max_per_task}" "${action_type}" \
    --batch_root "${batch_root}" \
    "${tasks_arg[@]}"
