#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${VENV_PATH}/bin/torchrun}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[ERROR] VENV_PATH not found: ${VENV_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${TORCHRUN_BIN}" && "${CONVERT_ONLY:-0}" != "1" ]]; then
  echo "[ERROR] torchrun executable not found: ${TORCHRUN_BIN}" >&2
  exit 1
fi

RAW_ROOT="${RAW_ROOT:-/path/to/robotwin_data}"
DATASET_NAME="${DATASET_NAME:-aloha-agilex_clean_50}"
TASKS="${TASKS:-}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-}"
FPS="${FPS:-50}"
CONVERTED_DATA_ROOT="${CONVERTED_DATA_ROOT:-${REPO_ROOT}/outputs/robotwin_spirit_dataset}"
OVERWRITE_DATASET="${OVERWRITE_DATASET:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
CONVERT_ONLY="${CONVERT_ONLY:-0}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/path/to/model_weights/Spirit-v1.5}"
if [[ "${CONVERT_ONLY}" != "1" ]]; then
  : "${PRETRAINED_PATH:?Please set PRETRAINED_PATH}"
fi

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "[ERROR] RAW_ROOT not found: ${RAW_ROOT}" >&2
  exit 1
fi

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  CONVERT_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_robotwin_to_spirit.py"
    --raw-root "${RAW_ROOT}"
    --output-root "${CONVERTED_DATA_ROOT}"
    --dataset-name "${DATASET_NAME}"
    --fps "${FPS}"
  )

  if [[ -n "${TASKS}" ]]; then
    CONVERT_CMD+=(--tasks "${TASKS}")
  fi

  if [[ -n "${MAX_EPISODES_PER_TASK}" ]]; then
    CONVERT_CMD+=(--max-episodes-per-task "${MAX_EPISODES_PER_TASK}")
  fi

  if [[ "${OVERWRITE_DATASET}" == "1" ]]; then
    CONVERT_CMD+=(--overwrite)
  fi

  echo "[INFO] Converting RobotWin data into Spirit format..."
  "${CONVERT_CMD[@]}"
else
  echo "[INFO] Skipping conversion and reusing existing dataset at ${CONVERTED_DATA_ROOT}"
  if [[ ! -f "${CONVERTED_DATA_ROOT}/meta/task_info.json" ]]; then
    echo "[ERROR] Converted dataset metadata not found: ${CONVERTED_DATA_ROOT}/meta/task_info.json" >&2
    exit 1
  fi
fi

if [[ "${CONVERT_ONLY}" == "1" ]]; then
  echo "[INFO] Conversion complete. CONVERT_ONLY=1, skipping training."
  exit 0
fi

export DATA_ROOT="${CONVERTED_DATA_ROOT}"
export VENV_PATH
export TORCHRUN_BIN

echo "[INFO] Starting finetuning with DATA_ROOT=${DATA_ROOT}"
exec bash "${REPO_ROOT}/scripts/run_finetune.sh"