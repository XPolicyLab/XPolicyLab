#!/bin/bash
# Wrapper around RoboOrchardLab/projects/holobrain/scripts/export.py.
#
# Stages the XPolicyLab-provided URDF from embodiments/arx_x5/ into the
# workspace before export so the pipeline embeds the correct kinematics asset.
#
# Usage:
#   bash export.sh [workspace_dir] [config_path]
# Defaults:
#   workspace_dir = ./workspace  (relative to RoboOrchardLab/projects/holobrain/)
#   config_path   = configs/config_holobrain_qwen_common.py

set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOBRAIN_PROJ_DIR="${POLICY_DIR}/RoboOrchardLab/projects/holobrain"
EMBODIMENTS_URDF_DIR="${POLICY_DIR}/embodiments/arx_x5"
DEFAULT_URDF="${EMBODIMENTS_URDF_DIR}/dual_x5_exact_from_x5a.urdf"

workspace="${1:-./workspace}"
config_path="${2:-configs/config_holobrain_qwen_common.py}"

cd "${HOLOBRAIN_PROJ_DIR}"

mkdir -p "${workspace}"
workspace_abs="$(cd "${workspace}" && pwd)"

if [[ ! -f "${DEFAULT_URDF}" ]]; then
    echo "[ERROR] URDF not found: ${DEFAULT_URDF}" >&2
    exit 1
fi

mkdir -p "${workspace_abs}/urdf/arx_x5"
cp "${DEFAULT_URDF}" "${workspace_abs}/urdf/arx_x5/dual_x5_exact_from_x5a.urdf"
if [[ -f "${EMBODIMENTS_URDF_DIR}/X5A.urdf" ]]; then
    cp "${EMBODIMENTS_URDF_DIR}/X5A.urdf" "${workspace_abs}/urdf/arx_x5/"
fi
echo "[INFO] Staged URDF from ${EMBODIMENTS_URDF_DIR} -> ${workspace_abs}/urdf/arx_x5/"

export XPOLICY_HOLOBRAIN_DATASETS="${XPOLICY_HOLOBRAIN_DATASETS:-robotwin2_0}"
export XPOLICY_HOLOBRAIN_URDF="${XPOLICY_HOLOBRAIN_URDF:-${DEFAULT_URDF}}"

echo "[INFO] Exporting to: ${workspace_abs}"
echo "[INFO] Config:       ${config_path}"
echo "[INFO] Datasets:     ${XPOLICY_HOLOBRAIN_DATASETS}"
echo "[INFO] URDF:         ${XPOLICY_HOLOBRAIN_URDF}"
python3 scripts/export.py --config "${config_path}" --workspace "${workspace_abs}"

echo ""
echo "[DONE] Exported model directory: ${workspace_abs}/model"
echo "       eval.sh resolves model_dir automatically from workspace/<6-tuple>/model"
echo "       inference_prefix: robotwin2_0"
