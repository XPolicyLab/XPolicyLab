# XPolicyLab deploy: policy server env=openvla_oft; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_ROOT="${POLICY_DIR}/openvla_oft"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${OPENVLA_CONDA_ENV:-openvla_oft}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${OPENVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10.6 -y
  fi
fi

conda activate "${CONDA_ENV}"
pip install torch torchvision torchaudio

cd "${OPENVLA_ROOT}"
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[OpenVLA_OFT] Done. conda activate ${CONDA_ENV}"
