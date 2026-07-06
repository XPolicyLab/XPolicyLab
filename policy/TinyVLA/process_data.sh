#!/bin/bash
set -e


if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  echo "       bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [raw_task_dirs]" >&2
  exit 1
fi

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
expert_data_num=${5:-}   # optional; empty = use all episodes
raw_task_dirs=${6:-}      # optional comma-separated task directory list

if [[ $# -eq 5 && -n "${expert_data_num}" && ! "${expert_data_num}" =~ ^[0-9]+$ ]]; then
  raw_task_dirs="${expert_data_num}"
  expert_data_num=""
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
SOURCE_ROOT="${XPL_SOURCE_ROOT:-${ROOT_DIR}/data/${bench_name}}"
WORKERS="${TINYVLA_PROCESS_WORKERS:-8}"
COMPRESSION="${TINYVLA_HDF5_COMPRESSION:-lzf}"

ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
out_dir="${POLICY_DIR}/data/${ckpt_setting}"

echo "[TinyVLA process_data] output: ${out_dir}"
echo "[TinyVLA process_data] source: ${SOURCE_ROOT}"
echo "[TinyVLA process_data] workers=${WORKERS}, compression=${COMPRESSION}"
if [[ -n "${raw_task_dirs}" ]]; then
  echo "[TinyVLA process_data] tasks=${raw_task_dirs}"
fi

#If the 4-tuple output directory already exists, let the user decide:
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

process_args=(
  "${POLICY_DIR}/process_data.py"
  "${bench_name}"
  "${ckpt_name}"
  "${env_cfg_type}"
  "${action_type}"
)
if [[ -n "${expert_data_num}" ]]; then
  process_args+=("${expert_data_num}")
fi
process_args+=(
  --source-root "${SOURCE_ROOT}"
  --output-dir "${out_dir}"
  --workers "${WORKERS}"
  --compression "${COMPRESSION}"
)
if [[ -n "${raw_task_dirs}" ]]; then
  process_args+=(--tasks "${raw_task_dirs}")
fi

python "${process_args[@]}"
