#!/bin/bash
set -euo pipefail

# ==================== 参数定义 ====================
usage="Usage: bash process_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type>"
dataset_name=${1:?${usage}}
task_name=${2:?${usage}}
env_cfg_type=${3:?${usage}}
expert_data_num=${4:?${usage}}
action_type=${5:?${usage}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/${dataset_name}/${task_name}/${env_cfg_type}"
OFFLINE_DIR="${SCRIPT_DIR}/RISE/policy_and_value/policy_offline_and_value"
CONVERTED_DATASET="${SCRIPT_DIR}/data/${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}-lerobot"

python "${SCRIPT_DIR}/RISE/process_data.py" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${expert_data_num}" \
    "${action_type}" \
    --data-dir "${DATA_DIR}"

echo "[RISE] Computing normalization stats for: ${CONVERTED_DATASET}"
cd "${OFFLINE_DIR}"
export PYTHONPATH="${OFFLINE_DIR}/src:${PYTHONPATH:-}"
RISE_XPOLICYLAB_DATASET="${CONVERTED_DATASET}" \
python scripts/compute_norm_stats_fast.py --config-name Compute_norm
