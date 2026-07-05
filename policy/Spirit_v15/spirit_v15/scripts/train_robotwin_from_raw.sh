#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${VENV_PATH}/bin/torchrun}"

RAW_ROOT="${1:?Usage: $0 <raw_root> <bench_name> <converted_data_root> <pretrained_path> <output_dir> [num_gpus] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode] [tasks_csv] [max_episodes_per_task] [fps] [overwrite_flag] [skip_convert] [convert_only]>}"
DATASET_NAME="${2:?Usage: $0 <raw_root> <bench_name> <converted_data_root> <pretrained_path> <output_dir> [num_gpus] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode] [tasks_csv] [max_episodes_per_task] [fps] [overwrite_flag] [skip_convert] [convert_only]>}"
CONVERTED_DATA_ROOT="${3:?Usage: $0 <raw_root> <bench_name> <converted_data_root> <pretrained_path> <output_dir> [num_gpus] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode] [tasks_csv] [max_episodes_per_task] [fps] [overwrite_flag] [skip_convert] [convert_only]>}"
PRETRAINED_PATH="${4:?Usage: $0 <raw_root> <bench_name> <converted_data_root> <pretrained_path> <output_dir> [num_gpus] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode] [tasks_csv] [max_episodes_per_task] [fps] [overwrite_flag] [skip_convert] [convert_only]>}"
OUTPUT_DIR="${5:?Usage: $0 <raw_root> <bench_name> <converted_data_root> <pretrained_path> <output_dir> [num_gpus] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode] [tasks_csv] [max_episodes_per_task] [fps] [overwrite_flag] [skip_convert] [convert_only]>}"
NUM_GPUS="${6:-1}"
BATCH_SIZE="${7:-32}"
MAX_TRAIN_STEPS="${8:-40000}"
LOG_INTERVAL="${9:-25}"
SAVE_STEPS="${10:-2500}"
NUM_WORKERS="${11:-4}"
PREFETCH_FACTOR="${12:-8}"
WANDB_MODE="${13:-disabled}"
TASKS_CSV="${14:-}"
MAX_EPISODES_PER_TASK="${15:-}"
FPS="${16:-50}"
OVERWRITE_FLAG="${17:-0}"
SKIP_CONVERT="${18:-0}"
CONVERT_ONLY="${19:-0}"

if [[ ! -x "${TORCHRUN_BIN}" && "${CONVERT_ONLY}" != "1" ]]; then
  echo "[ERROR] torchrun executable not found: ${TORCHRUN_BIN}" >&2
  exit 1
fi

if [[ ! -d "${PRETRAINED_PATH}" && "${CONVERT_ONLY}" != "1" ]]; then
  echo "[ERROR] PRETRAINED_PATH not found: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  bash "${REPO_ROOT}/scripts/prepare_robotwin_dataset.sh" \
    "${RAW_ROOT}" \
    "${DATASET_NAME}" \
    "${CONVERTED_DATA_ROOT}" \
    "${TASKS_CSV}" \
    "${MAX_EPISODES_PER_TASK}" \
    "${FPS}" \
    "${OVERWRITE_FLAG}"
else
  if [[ ! -f "${CONVERTED_DATA_ROOT}/meta/task_info.json" ]]; then
    echo "[ERROR] Converted dataset metadata not found: ${CONVERTED_DATA_ROOT}/meta/task_info.json" >&2
    exit 1
  fi
fi

if [[ "${CONVERT_ONLY}" == "1" ]]; then
  echo "[INFO] Conversion complete. CONVERT_ONLY=1, skipping training."
  exit 0
fi

if [[ ! -f "${PRETRAINED_PATH}/model.safetensors" ]]; then
  echo "[ERROR] model.safetensors not found in PRETRAINED_PATH: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_PATH}/config.json" ]]; then
  echo "[ERROR] config.json not found in PRETRAINED_PATH: ${PRETRAINED_PATH}" >&2
  exit 1
fi

echo "[INFO] Starting Spirit finetuning"
echo "[INFO] data_root=${CONVERTED_DATA_ROOT}"
echo "[INFO] pretrained_path=${PRETRAINED_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}"

exec "${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS}" \
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