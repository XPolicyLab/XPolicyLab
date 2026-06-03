#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${VENV_PATH}/bin/torchrun}"

CONVERTED_DATA_ROOT="${1:?Usage: $0 <converted_data_root> <pretrained_path> <output_dir> [gpu_ids_csv] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode]>}"
PRETRAINED_PATH="${2:?Usage: $0 <converted_data_root> <pretrained_path> <output_dir> [gpu_ids_csv] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode]>}"
OUTPUT_DIR="${3:?Usage: $0 <converted_data_root> <pretrained_path> <output_dir> [gpu_ids_csv] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode]>}"
GPU_ID="${4:-0}"
BATCH_SIZE="${5:-32}"
MAX_TRAIN_STEPS="${6:-40000}"
LOG_INTERVAL="${7:-25}"
SAVE_STEPS="${8:-2500}"
NUM_WORKERS="${9:-4}"
PREFETCH_FACTOR="${10:-8}"
WANDB_MODE="${11:-disabled}"
SEED="${12:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONHASHSEED="${SEED}"

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
echo "[INFO] seed=${SEED}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${SPIRIT_BACKBONE_PATH:-}" && -n "${PRETRAINED_PATH:-}" ]]; then
  _weights_root="$(dirname "${PRETRAINED_PATH}")"
  if [[ -d "${_weights_root}/Qwen3-VL-4B-Instruct" ]]; then
    export SPIRIT_BACKBONE_PATH="${_weights_root}/Qwen3-VL-4B-Instruct"
  fi
fi
if [[ -n "${SPIRIT_BACKBONE_PATH:-}" ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  echo "[INFO] spirit_backbone_path=${SPIRIT_BACKBONE_PATH}"
  echo "[INFO] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
fi

cd "${REPO_ROOT}"

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
  --wandb_mode "${WANDB_MODE}" \
  --seed "${SEED}"