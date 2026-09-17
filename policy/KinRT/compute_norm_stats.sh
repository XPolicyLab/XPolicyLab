#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
PYTHON_BIN="${KINRT_PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
train_config_name="${1:-${OPENPI_TRAIN_CONFIG_NAME:-kinrt_full_robodojo}}"
repo_id="${KINRT_ROBODOJO_REPO_ID:-RoboDojo_lerobot_v30_video}"
norm_stats_path="${OPENPI_ROOT}/assets/${train_config_name}/${repo_id}/norm_stats.json"

case "${train_config_name}" in
  kinrt_lora_robodojo|kinrt_full_robodojo) ;;
  *)
    echo "[KinRT][ERROR] Config must be kinrt_lora_robodojo or kinrt_full_robodojo." >&2
    exit 1
    ;;
esac
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[KinRT][ERROR] Policy environment not found. Run: bash ${POLICY_DIR}/install.sh" >&2
  exit 1
fi
if [[ -e "${norm_stats_path}" && "${KINRT_OVERWRITE_NORM_STATS:-0}" != "1" ]]; then
  echo "[KinRT][ERROR] Normalization statistics already exist: ${norm_stats_path}" >&2
  echo "[KinRT][ERROR] Set KINRT_OVERWRITE_NORM_STATS=1 only to recompute them intentionally." >&2
  exit 1
fi

echo "[KinRT] Computing new normalization statistics; use the published statistics to reproduce the Full35 delivery."
export KINRT_ROBODOJO_REPO_ID="${repo_id}"
cd "${OPENPI_ROOT}"
PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" scripts/compute_norm_stats.py --config-name "${train_config_name}"
