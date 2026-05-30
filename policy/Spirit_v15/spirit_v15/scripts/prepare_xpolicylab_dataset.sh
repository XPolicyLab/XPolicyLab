#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

RAW_DATA_ROOT="${1:?Usage: $0 <raw_data_root> <patterns_csv> <output_root> [task_name] [task_prompt] [fps|auto] [overwrite_flag] [max_episodes_per_target] [robot_type] [data_type] [data_version]>}"
PATTERNS_CSV="${2:?Usage: $0 <raw_data_root> <patterns_csv> <output_root> [task_name] [task_prompt] [fps|auto] [overwrite_flag] [max_episodes_per_target] [robot_type] [data_type] [data_version]>}"
OUTPUT_ROOT="${3:?Usage: $0 <raw_data_root> <patterns_csv> <output_root> [task_name] [task_prompt] [fps|auto] [overwrite_flag] [max_episodes_per_target] [robot_type] [data_type] [data_version]>}"
TASK_NAME="${4:-robodojo_multitask}"
TASK_PROMPT="${5:-Perform the instructed bimanual manipulation task.}"
FPS_RAW="${6:-auto}"
OVERWRITE_FLAG="${7:-0}"
MAX_EPISODES_PER_TARGET="${8:-}"
ROBOT_TYPE="${9:-aloha}"
DATA_TYPE="${10:-xspark}"
DATA_VERSION="${11:-v1.0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${RAW_DATA_ROOT}" ]]; then
  echo "[ERROR] RAW_DATA_ROOT not found: ${RAW_DATA_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a PATTERNS <<< "${PATTERNS_CSV}"
if [[ ${#PATTERNS[@]} -eq 0 ]]; then
  echo "[ERROR] No patterns provided in PATTERNS_CSV" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_xpolicylab_to_spirit.py"
  --output-root "${OUTPUT_ROOT}"
  --data-type "${DATA_TYPE}"
  --data-version "${DATA_VERSION}"
  --task-name "${TASK_NAME}"
  --task-prompt "${TASK_PROMPT}"
  --robot-type "${ROBOT_TYPE}"
)

for pattern in "${PATTERNS[@]}"; do
  if [[ -n "${pattern}" ]]; then
    CMD+=("${pattern}")
  fi
done

if [[ "${FPS_RAW}" != "auto" && -n "${FPS_RAW}" ]]; then
  CMD+=(--fps "${FPS_RAW}")
fi

if [[ "${OVERWRITE_FLAG}" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ -n "${MAX_EPISODES_PER_TARGET}" ]]; then
  CMD+=(--max-episodes-per-target "${MAX_EPISODES_PER_TARGET}")
fi

export XPOLICYLAB_DATA_ROOT="${RAW_DATA_ROOT}"

echo "[INFO] Converting XPolicyLab raw data to Spirit dataset format"
echo "[INFO] raw_data_root=${RAW_DATA_ROOT}"
echo "[INFO] patterns=${PATTERNS_CSV}"
echo "[INFO] output_root=${OUTPUT_ROOT}"

exec "${CMD[@]}"