#!/bin/bash
# Usage: bash process_data_batch.sh <dataset_name> <ckpt_name> <env_cfg_type> <action_type> \
#            <batch_root> [max_episodes_per_task] [tasks...]
set -euo pipefail

dataset_name=${1:?dataset_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
batch_root=${5:?batch_root required}
max_per_task=${6:-0}
shift $(( $# < 6 ? $# : 6 )) || true
tasks=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

ADAPTER_DIR="${SCRIPT_DIR}/GalaxeaVLA/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"
out_tag="$(xpolicylab_dataset_tag "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"

echo "[process_data_batch] root=${batch_root} -> data/${out_tag}-lerobot/"
echo "[process_data_batch] standardizing every camera frame to RGB HWC (240, 320, 3)"

tasks_arg=()
if [[ ${#tasks[@]} -gt 0 ]]; then
    tasks_arg=(--tasks "${tasks[@]}")
fi

source "${UPSTREAM_DIR}/.venv/bin/activate"
PYTHONPATH="${ROOT_DIR}:${UPSTREAM_DIR}/src:${PYTHONPATH:-}" \
python "${UPSTREAM_DIR}/xpolicylab_adapter/convert_to_galaxea_lerobot.py" \
    "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${max_per_task}" "${action_type}" \
    --batch_root "${batch_root}" \
    "${tasks_arg[@]}"
