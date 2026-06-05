# XPolicyLab deploy: policy server env=internvla_a1; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNVLA_ROOT="${POLICY_DIR}/internvla_a1"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${INTERNVLA_CONDA_ENV:-internvla_a1}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${INTERNVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
fi

conda activate "${CONDA_ENV}"

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf

cd "${INTERNVLA_ROOT}"
pip install -e .

TRANSFORMERS_DIR="${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/"
cp -r src/lerobot/policies/pi0/transformers_replace/models "${TRANSFORMERS_DIR}"
cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models "${TRANSFORMERS_DIR}"
cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models "${TRANSFORMERS_DIR}"

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[InternVLA_A1] Done. conda activate ${CONDA_ENV}"
