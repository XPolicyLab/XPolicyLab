#!/bin/bash
set -e


if [[ $# -ne 5 ]]; then
  echo "Usage: bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>" >&2
  exit 1
fi

dataset_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
SOURCE_ROOT="${XPL_SOURCE_ROOT:-${ROOT_DIR}/data/${dataset_name}}"
WORKERS="${TINYVLA_PROCESS_WORKERS:-8}"
COMPRESSION="${TINYVLA_HDF5_COMPRESSION:-lzf}"

ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
out_dir="${POLICY_DIR}/data/${ckpt_setting}"

echo "[TinyVLA process_data] output: ${out_dir}"
echo "[TinyVLA process_data] source: ${SOURCE_ROOT}"
echo "[TinyVLA process_data] workers=${WORKERS}, compression=${COMPRESSION}"

#If the 5-tuple output directory already exists, let the user decide:
#   - y  : skip processing entirely, reuse the existing dataset as-is
#   - N  : abort, the user must remove the directory manually before rerunning
if [[ -d "${out_dir}" ]]; then
  echo "[TinyVLA process_data] dataset already exists: ${out_dir}"
  read -r -p "Skip processing and reuse the existing dataset? [y/N]: " ans
  case "${ans}" in
    [yY]|[yY][eE][sS])
      echo "[TinyVLA process_data] skipping; reusing existing dataset."
      exit 0
      ;;
    *)
      echo "[TinyVLA process_data] aborting. Remove ${out_dir} manually and rerun." >&2
      exit 1
      ;;
  esac
fi

python "${POLICY_DIR}/process_data.py" \
  "${dataset_name}" \
  "${ckpt_name}" \
  "${env_cfg_type}" \
  "${expert_data_num}" \
  "${action_type}" \
  --source-root "${SOURCE_ROOT}" \
  --output-dir "${out_dir}" \
  --workers "${WORKERS}" \
  --compression "${COMPRESSION}"
