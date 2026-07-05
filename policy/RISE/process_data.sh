#!/bin/bash
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# expert_data_num: optional; empty = use all episodes.
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:-joint}
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/${bench_name}/${ckpt_name}/${env_cfg_type}"
OFFLINE_DIR="${SCRIPT_DIR}/RISE/policy_and_value/policy_offline_and_value"
ADAPTER_DIR="${SCRIPT_DIR}/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"
out_tag="$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
CONVERTED_DATASET="${SCRIPT_DIR}/data/${out_tag}-lerobot"

echo "[process_data] ${bench_name}/${ckpt_name}/${env_cfg_type} x${expert_data_num:-all} (${action_type}) -> data/${out_tag}-lerobot/"

py_args=(
    "${bench_name}"
    "${ckpt_name}"
    "${env_cfg_type}"
    "${action_type}"
    --data-dir "${DATA_DIR}"
)
if [[ -n "${expert_data_num}" ]]; then
    py_args+=(--expert-data-num "${expert_data_num}")
fi
python "${SCRIPT_DIR}/RISE/process_data.py" "${py_args[@]}"

echo "[RISE] Computing normalization stats for: ${CONVERTED_DATASET}"
cd "${OFFLINE_DIR}"
export PYTHONPATH="${OFFLINE_DIR}/src:${PYTHONPATH:-}"
RISE_XPOLICYLAB_DATASET="${CONVERTED_DATASET}" \
python scripts/compute_norm_stats_fast.py --config-name Compute_norm
