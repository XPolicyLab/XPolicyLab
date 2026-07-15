#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${MIBOT_CONDA_ENV:-mibot}"

echo "[Xiaomi_Robotics_1] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"
echo "[Xiaomi_Robotics_1] CONDA_ENV=${CONDA_ENV}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please install Miniconda/Anaconda first." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.12 -y
fi
conda activate "${CONDA_ENV}"

# Core dependencies
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1 scipy numpy Pillow ninja
pip install flash-attn==2.8.3 --no-build-isolation

echo "[Xiaomi_Robotics_1] Installation finished."
echo "[Xiaomi_Robotics_1] Activate env: conda activate ${CONDA_ENV}"
