#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${VENV_PATH}/bin/torchrun}"
DEFAULT_RAW_DATA_ROOT="/mnt/nfs/niantian/RoboDojo_env/data"
DEFAULT_PATTERNS_CSV="RoboDojo_real.*.*"
DEFAULT_CONVERTED_DATA_ROOT="/mnt/xspark-data/xspark_shared/spirit_datasets/robodojo_real_piper_cotrain"
DEFAULT_PRETRAINED_PATH="/mnt/xspark-data/xspark_shared/model_weights/Spirit-v1.5"
DEFAULT_OUTPUT_DIR="/mnt/xspark-data/xspark_shared/train_outputs/spirit_robodojo_real_piper_cotrain"
DEFAULT_GPU_IDS="0,1,2,3"
DEFAULT_MASTER_PORT="29501"

USAGE="Usage: $0 [raw_data_root=${DEFAULT_RAW_DATA_ROOT}] [patterns_csv=${DEFAULT_PATTERNS_CSV}] [converted_data_root=${DEFAULT_CONVERTED_DATA_ROOT}] [pretrained_path=${DEFAULT_PRETRAINED_PATH}] [output_dir=${DEFAULT_OUTPUT_DIR}] [gpu_ids_csv=${DEFAULT_GPU_IDS}] [batch_size] [max_train_steps] [log_interval] [save_steps] [num_workers] [prefetch_factor] [wandb_mode] [task_name] [task_prompt] [fps|auto] [overwrite_flag] [max_episodes_per_target] [robot_type] [data_type] [data_version] [skip_convert] [convert_only] [master_port=${DEFAULT_MASTER_PORT}]"

if [[ $# != 0 && $# < 3 ]]; then
  echo "${USAGE}" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  RAW_DATA_ROOT="${DEFAULT_RAW_DATA_ROOT}"
  PATTERNS_CSV="${DEFAULT_PATTERNS_CSV}"
  CONVERTED_DATA_ROOT="${DEFAULT_CONVERTED_DATA_ROOT}"
  PRETRAINED_PATH="${DEFAULT_PRETRAINED_PATH}"
  OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
  GPU_IDS="${DEFAULT_GPU_IDS}"
  BATCH_SIZE="32"
  MAX_TRAIN_STEPS="40000"
  LOG_INTERVAL="25"
  SAVE_STEPS="2500"
  NUM_WORKERS="4"
  PREFETCH_FACTOR="8"
  WANDB_MODE="disabled"
  TASK_NAME="robodojo_cotrain"
  TASK_PROMPT="Perform the instructed bimanual manipulation task."
  FPS_RAW="auto"
  OVERWRITE_FLAG="0"
  MAX_EPISODES_PER_TARGET=""
  ROBOT_TYPE="aloha"
  DATA_TYPE="RoboDojo"
  DATA_VERSION="v1.0"
  SKIP_CONVERT="0"
  CONVERT_ONLY="0"
  MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
elif [[ $# -ge 5 ]]; then
  RAW_DATA_ROOT="${1}"
  PATTERNS_CSV="${2}"
  CONVERTED_DATA_ROOT="${3}"
  PRETRAINED_PATH="${4}"
  OUTPUT_DIR="${5}"
  GPU_IDS="${6:-${DEFAULT_GPU_IDS}}"
  BATCH_SIZE="${7:-32}"
  MAX_TRAIN_STEPS="${8:-40000}"
  LOG_INTERVAL="${9:-25}"
  SAVE_STEPS="${10:-2500}"
  NUM_WORKERS="${11:-4}"
  PREFETCH_FACTOR="${12:-8}"
  WANDB_MODE="${13:-disabled}"
  TASK_NAME="${14:-robodojo_cotrain}"
  TASK_PROMPT="${15:-Perform the instructed bimanual manipulation task.}"
  FPS_RAW="${16:-auto}"
  OVERWRITE_FLAG="${17:-0}"
  MAX_EPISODES_PER_TARGET="${18:-}"
  ROBOT_TYPE="${19:-aloha}"
  DATA_TYPE="${20:-RoboDojo}"
  DATA_VERSION="${21:-v1.0}"
  SKIP_CONVERT="${22:-0}"
  CONVERT_ONLY="${23:-0}"
  MASTER_PORT="${MASTER_PORT:-${24:-${DEFAULT_MASTER_PORT}}}"
else
  RAW_DATA_ROOT="${DEFAULT_RAW_DATA_ROOT}"
  PATTERNS_CSV="${DEFAULT_PATTERNS_CSV}"
  CONVERTED_DATA_ROOT="${1}"
  PRETRAINED_PATH="${2}"
  OUTPUT_DIR="${3}"
  GPU_IDS="${4:-${DEFAULT_GPU_IDS}}"
  BATCH_SIZE="${5:-32}"
  MAX_TRAIN_STEPS="${6:-40000}"
  LOG_INTERVAL="${7:-25}"
  SAVE_STEPS="${8:-2500}"
  NUM_WORKERS="${9:-4}"
  PREFETCH_FACTOR="${10:-8}"
  WANDB_MODE="${11:-disabled}"
  TASK_NAME="${12:-robodojo_cotrain}"
  TASK_PROMPT="${13:-Perform the instructed bimanual manipulation task.}"
  FPS_RAW="${14:-auto}"
  OVERWRITE_FLAG="${15:-0}"
  MAX_EPISODES_PER_TARGET="${16:-}"
  ROBOT_TYPE="${17:-aloha}"
  DATA_TYPE="${18:-RoboDojo}"
  DATA_VERSION="${19:-v1.0}"
  SKIP_CONVERT="${20:-0}"
  CONVERT_ONLY="${21:-0}"
  MASTER_PORT="${MASTER_PORT:-${22:-${DEFAULT_MASTER_PORT}}}"
fi

GPU_IDS="${GPU_IDS// /}"
if [[ -z "${GPU_IDS}" ]]; then
  echo "[ERROR] GPU ID list is empty. Pass values like 0 or 4,5,6,7" >&2
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
  echo "[ERROR] Failed to parse GPU IDs from: ${GPU_IDS}" >&2
  exit 1
fi

if [[ ! -x "${TORCHRUN_BIN}" && "${CONVERT_ONLY}" != "1" ]]; then
  echo "[ERROR] torchrun executable not found: ${TORCHRUN_BIN}" >&2
  exit 1
fi

if [[ ! -d "${PRETRAINED_PATH}" && "${CONVERT_ONLY}" != "1" ]]; then
  echo "[ERROR] PRETRAINED_PATH not found: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  bash "${REPO_ROOT}/scripts/prepare_xpolicylab_dataset.sh" \
    "${RAW_DATA_ROOT}" \
    "${PATTERNS_CSV}" \
    "${CONVERTED_DATA_ROOT}" \
    "${TASK_NAME}" \
    "${TASK_PROMPT}" \
    "${FPS_RAW}" \
    "${OVERWRITE_FLAG}" \
    "${MAX_EPISODES_PER_TARGET}" \
    "${ROBOT_TYPE}" \
    "${DATA_TYPE}" \
    "${DATA_VERSION}"
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
echo "[INFO] raw_data_root=${RAW_DATA_ROOT}"
echo "[INFO] patterns=${PATTERNS_CSV}"
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