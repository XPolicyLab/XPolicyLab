#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
fifth_arg=${5:-}
sixth_arg=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
PYTHON_BIN="${KINRT_PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
repo_id="${KINRT_ROBODOJO_REPO_ID:-RoboDojo-KinRT-${env_cfg_type}-${action_type}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[KinRT][ERROR] Policy environment not found. Run: bash ${POLICY_DIR}/install.sh" >&2
  exit 1
fi

python_args=(
  "${bench_name}"
  "${ckpt_name}"
  "${env_cfg_type}"
  "${action_type}"
  --repo-id "${repo_id}"
  --mode "${OPENPI_DATA_MODE:-image}"
  --metadata-fps "${KINRT_LEROBOT_METADATA_FPS:-50}"
  --image-writer-processes "${KINRT_IMAGE_WRITER_PROCESSES:-0}"
  --image-writer-threads "${KINRT_IMAGE_WRITER_THREADS:-4}"
)
if [[ -n "${fifth_arg}" ]]; then
  python_args+=("${fifth_arg}")
  if [[ -n "${sixth_arg}" ]]; then
    python_args+=("${sixth_arg}")
  fi
fi
if [[ -n "${KINRT_ROBODOJO_SOURCE_DIR:-}" ]]; then
  python_args+=(--source-dir "${KINRT_ROBODOJO_SOURCE_DIR}")
fi
if [[ "${KINRT_OVERWRITE_DATASET:-0}" == "1" ]]; then
  python_args+=(--overwrite)
fi

export KINRT_ROBODOJO_REPO_ID="${repo_id}"
PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${POLICY_DIR}/process_data.py" "${python_args[@]}"
