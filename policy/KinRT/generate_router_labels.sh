#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
PYTHON_BIN="${KINRT_PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
repo_id="${KINRT_ROBODOJO_REPO_ID:-RoboDojo-KinRT-arx_x5-joint}"
hf_lerobot_home="${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}"
repo_root="${1:-${hf_lerobot_home}/${repo_id}}"
output_dir="${2:-${repo_root}/meta/router_labels_k4}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[KinRT][ERROR] Policy environment not found. Run: bash ${POLICY_DIR}/install.sh" >&2
  exit 1
fi
if [[ ! -d "${repo_root}/meta" ]]; then
  echo "[KinRT][ERROR] LeRobot dataset not found: ${repo_root}" >&2
  exit 1
fi

PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/generate_router_labels.py" \
  --repo-root "${repo_root}" \
  --output-dir "${output_dir}" \
  --action-horizon 50 \
  --feature-mode chunk_velocity \
  --pca-components 64 \
  --num-clusters 4 \
  --seed "${KINRT_ROUTER_SEED:-0}"
