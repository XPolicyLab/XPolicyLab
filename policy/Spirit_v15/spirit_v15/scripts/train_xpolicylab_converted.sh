#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${VENV_PATH}/bin/torchrun}"
DEFAULT_CONVERTED_DATA_ROOT="/mnt/xspark-data/xspark_shared/spirit_datasets/robodojo_real_piper_cotrain/"
DEFAULT_PRETRAINED_PATH="/mnt/xspark-data/xspark_shared/model_weights/Spirit-v1.5"
DEFAULT_OUTPUT_DIR="/mnt/xspark-data/xspark_shared/train_outputs/spirit_robodojo_real_piper_cotrain"
DEFAULT_GPU_IDS="0,1,2,3"
DEFAULT_MASTER_PORT="29502"

USAGE="Usage: $0 [converted_data_root=${DEFAULT_CONVERTED_DATA_ROOT}] [pretrained_path=${DEFAULT_PRETRAINED_PATH}] [output_dir=${DEFAULT_OUTPUT_DIR}] [gpu_ids_csv=${DEFAULT_GPU_IDS}] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode] [master_port=${DEFAULT_MASTER_PORT}]"

if [[ $# != 0 && $# < 3 ]]; then
  echo "${USAGE}" >&2
  exit 1
fi

CONVERTED_DATA_ROOT="${1:-${DEFAULT_CONVERTED_DATA_ROOT}}"
PRETRAINED_PATH="${2:-${DEFAULT_PRETRAINED_PATH}}"
OUTPUT_DIR="${3:-${DEFAULT_OUTPUT_DIR}}"
GPU_ID="${4:-${DEFAULT_GPU_IDS}}"
BATCH_SIZE="${5:-32}"
MAX_TRAIN_STEPS="${6:-40000}"
LOG_INTERVAL="${7:-25}"
SAVE_STEPS="${8:-2500}"
NUM_WORKERS="${9:-4}"
PREFETCH_FACTOR="${10:-8}"
WANDB_MODE="${11:-disabled}"
MASTER_PORT="${MASTER_PORT:-${12:-${DEFAULT_MASTER_PORT}}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

GPU_IDS="${GPU_ID// /}"
if [[ -z "${GPU_IDS}" ]]; then
  echo "[ERROR] GPU ID list is empty. Pass values like 0 or 0,1,2,3" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=0
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
  if [[ -n "${gpu_id}" ]]; then
    NUM_GPUS=$((NUM_GPUS + 1))
  fi
done

if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "[ERROR] Failed to parse GPU IDs from: ${GPU_ID}" >&2
  exit 1
fi

if [[ ! -d "${CONVERTED_DATA_ROOT}" ]]; then
  echo "[ERROR] CONVERTED_DATA_ROOT not found: ${CONVERTED_DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CONVERTED_DATA_ROOT}/meta/task_info.json" ]]; then
  echo "[ERROR] Converted dataset metadata not found: ${CONVERTED_DATA_ROOT}/meta/task_info.json" >&2
  exit 1
fi

if [[ ! -d "${PRETRAINED_PATH}" ]]; then
  echo "[ERROR] PRETRAINED_PATH not found: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_PATH}/model.safetensors" ]]; then
  echo "[ERROR] model.safetensors not found in PRETRAINED_PATH: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_PATH}/config.json" ]]; then
  echo "[ERROR] config.json not found in PRETRAINED_PATH: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "[ERROR] torchrun executable not found: ${TORCHRUN_BIN}" >&2
  exit 1
fi

echo "[INFO] Starting Spirit finetuning from converted dataset"
echo "[INFO] data_root=${CONVERTED_DATA_ROOT}"
echo "[INFO] pretrained_path=${PRETRAINED_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] gpu_ids=${GPU_IDS}"
echo "[INFO] num_gpus=${NUM_GPUS}"
echo "[INFO] master_port=${MASTER_PORT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

exec "${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/train.py" \
  --data_root "${CONVERTED_DATA_ROOT}" \
  --pretrained_path "${PRETRAINED_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --save_steps "${SAVE_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --wandb_mode "${WANDB_MODE}"