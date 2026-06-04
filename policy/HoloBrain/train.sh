#!/bin/bash
set -euo pipefail

# XPolicyLab train.sh — 8 参数（见 XPolicyLab/README.md §3）
#   bash train.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> \
#                 <expert_data_num> <action_type> <seed> <gpu_id>
#
# 命名约定（README §命名约定）：
#   数据集 5 元组 → data/<dataset>-<task_name>-<env>-<num>-<action>/
#   训练 6 元组   → workspace/<dataset>-<ckpt_name>-<env>-<num>-<action>-<seed>/
# 常规训练令 ckpt_name=task_name；cotrain 时 ckpt_name 可与 task_name 不同，
# 此时 process_data.sh 的 task_name 参数应填 cotrain 等数据标识。

dataset_name=${1:?dataset_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
expert_data_num=${5:?expert_data_num required}
action_type=${6:?action_type required}
seed=${7:?seed required}
gpu_id=${8:?gpu_id required}
config_path="${HOLOBRAIN_CONFIG:-configs/config_holobrain_qwen_common.py}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOBRAIN_PROJ_DIR="${POLICY_DIR}/RoboOrchardLab/projects/holobrain"

# ---- paths derived from train.sh args (README naming) ----
DATA_RUN_ID="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
CKPT_RUN_ID="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
DATA_DIR="${POLICY_DIR}/data/${DATA_RUN_ID}"
LMDB_DIR="${DATA_DIR}/lmdb"
WORKSPACE_DIR="${POLICY_DIR}/workspace/${CKPT_RUN_ID}"

# cotrain: LMDB may be keyed by ckpt_name instead of task_name
if [[ ! -d "${LMDB_DIR}" && "${ckpt_name}" != "${task_name}" ]]; then
    ALT_DATA_RUN_ID="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
    ALT_LMDB_DIR="${POLICY_DIR}/data/${ALT_DATA_RUN_ID}/lmdb"
    if [[ -d "${ALT_LMDB_DIR}" ]]; then
        DATA_RUN_ID="${ALT_DATA_RUN_ID}"
        DATA_DIR="${POLICY_DIR}/data/${DATA_RUN_ID}"
        LMDB_DIR="${ALT_LMDB_DIR}"
        echo "[INFO] Using ckpt_name-keyed dataset: ${DATA_RUN_ID}"
    fi
fi

DEFAULT_URDF="${POLICY_DIR}/embodiments/arx_x5/dual_x5_exact_from_x5a.urdf"
export XPOLICY_HOLOBRAIN_URDF="${XPOLICY_HOLOBRAIN_URDF:-${DEFAULT_URDF}}"

if [[ ! -f "${XPOLICY_HOLOBRAIN_URDF}" ]]; then
    echo "[ERROR] URDF not found: ${XPOLICY_HOLOBRAIN_URDF}" >&2
    exit 1
fi

if [[ ! -d "${LMDB_DIR}" ]]; then
    echo "[ERROR] LMDB not found: ${LMDB_DIR}" >&2
    echo "        Expected data_run_id=${DATA_RUN_ID} (5-tuple from train args)" >&2
    echo "        Run: bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}" >&2
    if [[ "${ckpt_name}" != "${task_name}" ]]; then
        echo "        Or (cotrain): bash process_data.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type}" >&2
    fi
    exit 1
fi

CKPT_DIR="${WORKSPACE_DIR}/checkpoints"
if [[ -d "${CKPT_DIR}" ]] && [[ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]]; then
    echo "[ERROR] Workspace already has checkpoints: ${CKPT_DIR}" >&2
    echo "        ckpt_run_id=${CKPT_RUN_ID}" >&2
    echo "        Fix: use a different <seed>, or rm -rf '${WORKSPACE_DIR}'" >&2
    exit 1
fi
mkdir -p "${WORKSPACE_DIR}"

export XPOLICY_HOLOBRAIN_LMDB="${LMDB_DIR}"
export XPOLICY_HOLOBRAIN_DATASETS="robotwin2_0"

NUM_GPUS=$(echo "${gpu_id}" | tr ',' '\n' | grep -c .)
PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

cd "${HOLOBRAIN_PROJ_DIR}"
echo "[INFO] data_run_id=${DATA_RUN_ID}"
echo "[INFO] ckpt_run_id=${CKPT_RUN_ID}"
echo "[INFO] dataset=${dataset_name} task=${task_name} ckpt=${ckpt_name} env=${env_cfg_type} num=${expert_data_num} action=${action_type} seed=${seed}"
echo "[INFO] LMDB:      ${LMDB_DIR}"
echo "[INFO] Workspace: ${WORKSPACE_DIR}"
echo "[INFO] URDF:      ${XPOLICY_HOLOBRAIN_URDF}"
echo "[INFO] Config:    ${config_path}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (count=${NUM_GPUS})"
echo "[INFO] BS:        ${XPOLICY_HOLOBRAIN_BATCH_SIZE:-16} (per-process; effective=${NUM_GPUS}×bs)"
echo "[INFO] Precision: ${MIXED_PRECISION}"

if [[ "${NUM_GPUS}" -le 1 ]]; then
    echo "[INFO] Mode:      single-GPU (accelerate launch, port=${PORT})"
    accelerate launch \
        --num_processes 1 \
        --mixed_precision "${MIXED_PRECISION}" \
        --main_process_port "${PORT}" \
        scripts/train.py \
        --workspace "${WORKSPACE_DIR}" \
        --config "${config_path}"
else
    echo "[INFO] Mode:      multi-GPU (accelerate launch, ${NUM_GPUS} processes, port=${PORT})"
    accelerate launch \
        --num_machines 1 \
        --num_processes "${NUM_GPUS}" \
        --multi_gpu \
        --mixed_precision "${MIXED_PRECISION}" \
        --main_process_port "${PORT}" \
        scripts/train.py \
        --workspace "${WORKSPACE_DIR}" \
        --config "${config_path}"
fi
