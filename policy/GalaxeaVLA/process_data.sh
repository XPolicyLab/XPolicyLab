#!/bin/bash
# Usage: bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> [action_type]
set -euo pipefail

dataset_name=${1:?dataset_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
expert_data_num=${4:?expert_data_num required}
action_type=${5:-joint}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

ADAPTER_DIR="${SCRIPT_DIR}/GalaxeaVLA/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"
out_tag="$(xpolicylab_dataset_tag "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"

echo "[process_data] ${dataset_name}/${ckpt_name}/${env_cfg_type} x${expert_data_num} (${action_type}) -> data/${out_tag}-lerobot/"

source "${UPSTREAM_DIR}/.venv/bin/activate"
PYTHONPATH="${ROOT_DIR}:${UPSTREAM_DIR}/src:${PYTHONPATH:-}" \
python "${UPSTREAM_DIR}/xpolicylab_adapter/convert_to_galaxea_lerobot.py" \
    "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}"
