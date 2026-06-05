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
DATA_DIR="${ROOT_DIR}/data/${dataset_name}/${ckpt_name}/${env_cfg_type}"
OFFLINE_DIR="${SCRIPT_DIR}/RISE/policy_and_value/policy_offline_and_value"
ADAPTER_DIR="${SCRIPT_DIR}/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"
out_tag="$(xpolicylab_dataset_tag "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
CONVERTED_DATASET="${SCRIPT_DIR}/data/${out_tag}-lerobot"

echo "[process_data] ${dataset_name}/${ckpt_name}/${env_cfg_type} x${expert_data_num} (${action_type}) -> data/${out_tag}-lerobot/"

python "${SCRIPT_DIR}/RISE/process_data.py" \
    "${dataset_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    --data-dir "${DATA_DIR}"

echo "[RISE] Computing normalization stats for: ${CONVERTED_DATASET}"
cd "${OFFLINE_DIR}"
export PYTHONPATH="${OFFLINE_DIR}/src:${PYTHONPATH:-}"
RISE_XPOLICYLAB_DATASET="${CONVERTED_DATASET}" \
python scripts/compute_norm_stats_fast.py --config-name Compute_norm
