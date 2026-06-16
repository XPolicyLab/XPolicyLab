# XPolicyLab deploy: policy server env=XVLA; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XVLA_ROOT="${POLICY_DIR}/xvla"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${XVLA_CONDA_ENV:-XVLA}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${XVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
fi

conda activate "${CONDA_ENV}"

cd "${XVLA_ROOT}"
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[X_VLA] Done. conda activate ${CONDA_ENV}"
