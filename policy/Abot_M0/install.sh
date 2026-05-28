#!/usr/bin/env bash
# Abot 上游需手动 clone ABot-Manipulation；本脚本仅安装 XPolicyLab 并检查 abot_m0 目录。
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABOT_ROOT="${POLICY_DIR}/abot_m0"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

if [[ ! -d "${ABOT_ROOT}" ]]; then
  echo "[Abot_M0] abot_m0/ not found. See abot_m0/INSTALLATION.md for upstream clone steps." >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1 && [[ -d "${ABOT_ROOT}/requirements.txt" || -f "${ABOT_ROOT}/requirements.txt" ]]; then
  echo "[Abot_M0] Install ABot conda env per abot_m0/INSTALLATION.md (not automated here)."
fi

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[Abot_M0] XPolicyLab installed."
echo "[Abot_M0] Next: follow abot_m0/INSTALLATION.md for ABot-Manipulation + vggt setup."
