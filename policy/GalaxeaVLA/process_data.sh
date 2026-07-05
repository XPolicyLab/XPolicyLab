#!/bin/bash
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
expert_data_num=${5:-}   # optional; empty = use all episodes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

ADAPTER_DIR="${SCRIPT_DIR}/GalaxeaVLA/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"
out_tag="$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"

if [[ -n "${expert_data_num}" ]]; then
    echo "[process_data] ${bench_name}/${ckpt_name}/${env_cfg_type} x${expert_data_num} (${action_type}) -> data/${out_tag}-lerobot/"
else
    echo "[process_data] ${bench_name}/${ckpt_name}/${env_cfg_type} (all episodes, ${action_type}) -> data/${out_tag}-lerobot/"
fi

source "${UPSTREAM_DIR}/.venv/bin/activate"
PYTHONPATH="${ROOT_DIR}:${UPSTREAM_DIR}/src:${PYTHONPATH:-}" \
python "${UPSTREAM_DIR}/xpolicylab_adapter/convert_to_galaxea_lerobot.py" \
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
    ${expert_data_num:+"${expert_data_num}"}
