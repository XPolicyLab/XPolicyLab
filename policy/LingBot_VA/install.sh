# XPolicyLab deploy: policy server env=lingbot_va; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_ROOT="${POLICY_DIR}/lingbot_va"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${LINGBOT_VA_CONDA_ENV:-lingbot_va}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${LINGBOT_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10.6 -y
  fi
fi

conda activate "${CONDA_ENV}"

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install lerobot==0.3.3 scipy wandb --no-deps

cd "${LINGBOT_ROOT}"
pip install -e .

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[LingBot_VA] Done. conda activate ${CONDA_ENV}"
