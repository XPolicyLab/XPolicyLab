#!/usr/bin/env bash
# XPolicyLab deploy: policy server env=motus; run setup_eval_policy_server.sh with this env.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTUS_ROOT="${POLICY_DIR}/motus"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
MOTUS_CONDA_ENV="${MOTUS_CONDA_ENV:-motus}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${MOTUS_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${MOTUS_CONDA_ENV}"; then
    conda create -n "${MOTUS_CONDA_ENV}" python=3.10 -y
  fi
fi

conda activate "${MOTUS_CONDA_ENV}"

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install packaging psutil ninja wheel
pip install flash-attn --no-build-isolation

cd "${MOTUS_ROOT}"
pip install -r requirements.txt
pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt
pip install -e .

cd "${XPOLICYLAB_ROOT}"
pip install -e ".[robodojo]"

echo "[Motus] Installation finished. conda activate ${MOTUS_CONDA_ENV}"
