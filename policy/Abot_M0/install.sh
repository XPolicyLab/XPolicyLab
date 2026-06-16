# XPolicyLab deploy: policy server env=ABot; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
# Abot 上游需手动 clone ABot-Manipulation；本脚本在 ABot conda 中安装 XPolicyLab 依赖。
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABOT_ROOT="${POLICY_DIR}/abot_m0"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
ABOT_CONDA_ENV="${ABOT_CONDA_ENV:-ABot}"

if [[ ! -d "${ABOT_ROOT}" ]]; then
  echo "[Abot_M0] abot_m0/ not found. See abot_m0/INSTALLATION.md for upstream clone steps." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[Abot_M0] conda not found. Install ABot env manually per abot_m0/INSTALLATION.md." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ABOT_CONDA_ENV}"; then
  echo "[Abot_M0] conda env '${ABOT_CONDA_ENV}' not found." >&2
  echo "[Abot_M0] Create it per abot_m0/INSTALLATION.md, then re-run: bash install.sh" >&2
  exit 1
fi

conda activate "${ABOT_CONDA_ENV}"

cd "${XPOLICYLAB_ROOT}"
pip install -e .
pip install h5py opencv-python pyyaml

python -c "import XPolicyLab; import cv2, h5py, yaml; print('[Abot_M0] XPolicyLab + deps ok in', '${ABOT_CONDA_ENV}')"

echo "[Abot_M0] XPolicyLab installed in conda env: ${ABOT_CONDA_ENV}"
echo "[Abot_M0] Next: ensure ABot-Manipulation + vggt per abot_m0/INSTALLATION.md"
