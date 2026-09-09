#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
PYTHON_BIN="${KINRT_PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
repo_id="${KINRT_ROBODOJO_REPO_ID:-RoboDojo_lerobot_v30_video}"
hf_lerobot_home="${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}"
repo_root="${1:-${hf_lerobot_home}/${repo_id}}"
router_labels_subdir=router_labels_k4
if [[ "${repo_id}" == "RoboDojo_lerobot_v30_video" ]]; then
  router_labels_subdir=router_labels_k4_full35
fi
output_dir="${2:-${repo_root}/meta/${router_labels_subdir}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[KinRT][ERROR] Policy environment not found. Run: bash ${POLICY_DIR}/install.sh" >&2
  exit 1
fi
if [[ ! -d "${repo_root}/meta" ]]; then
  echo "[KinRT][ERROR] LeRobot dataset not found: ${repo_root}" >&2
  exit 1
fi
if [[ -e "${output_dir}/router_labels.npy" && "${KINRT_OVERWRITE_ROUTER_LABELS:-0}" != "1" ]]; then
  echo "[KinRT][ERROR] Router labels already exist: ${output_dir}/router_labels.npy" >&2
  echo "[KinRT][ERROR] Use another output directory, or set KINRT_OVERWRITE_ROUTER_LABELS=1 to regenerate intentionally." >&2
  exit 1
fi

echo "[KinRT] Generating labels for the current dataset frame order; these are not the published Full35 assets."
echo "[KinRT] Use a separate KINRT_ROBODOJO_REPO_ID for converted or reordered datasets."

PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/generate_router_labels.py" \
  --repo-root "${repo_root}" \
  --output-dir "${output_dir}" \
  --action-horizon 50 \
  --feature-mode chunk_velocity \
  --pca-components 64 \
  --num-clusters 4 \
  --seed "${KINRT_ROUTER_SEED:-0}"
