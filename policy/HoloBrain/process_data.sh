#!/bin/bash
set -euo pipefail

# XPolicyLab process_data.sh — 5 参数（见 XPolicyLab/README.md §2）
#   bash process_data.sh <dataset_name> <task_name> <env_cfg_type> \
#                        <expert_data_num> <action_type>
#
# 输出目录（5 元组，须与 train.sh 的 data_run_id 一致）：
#   policy/HoloBrain/data/<dataset>-<task_name>-<env>-<num>-<action>/

dataset_name=${1:?dataset_name required}
task_name=${2:?task_name required}
env_cfg_type=${3:?env_cfg_type required}
expert_data_num=${4:?expert_data_num required}
action_type=${5:?action_type required}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/HoloBrain"
export XPOLICY_HOLOBRAIN_URDF="${XPOLICY_HOLOBRAIN_URDF:-${POLICY_DIR}/embodiments/arx_x5/dual_x5_exact_from_x5a.urdf}"

DATA_RUN_ID="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
input_dir="${ROOT_DIR}/data/${dataset_name}/${task_name}/${env_cfg_type}"
output_dir="${POLICY_DIR}/data/${DATA_RUN_ID}"

if [[ ! -d "${input_dir}/data" ]]; then
    echo "[ERROR] HDF5 input not found: ${input_dir}/data" >&2
    exit 1
fi

echo "[INFO] data_run_id=${DATA_RUN_ID}"
echo "[INFO] input:  ${input_dir}/data"
echo "[INFO] output: ${output_dir}"

python "${POLICY_DIR}/RoboOrchardLab/projects/holobrain/process_data.py" \
    --project-root "${ROOT_DIR}" \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --task-name "${task_name}" \
    --env-cfg-type "${env_cfg_type}" \
    --expert-data-num "${expert_data_num}" \
    --action-type "${action_type}" \
    --overwrite

python3 -m robo_orchard_lab.dataset.robotwin.robotwin_packer \
    --input_path "${output_dir}/robotwin_packer_input" \
    --output_path "${output_dir}/lmdb" \
    --task_names "${task_name}" \
    --config_name demo_clean

echo "[INFO] LMDB ready: ${output_dir}/lmdb"
echo "[INFO] train with matching args, e.g.:"
echo "       bash train.sh ${dataset_name} ${task_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} 0 <gpu_id>"
