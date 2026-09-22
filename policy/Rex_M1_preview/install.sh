#!/usr/bin/env bash
# Adapted for Rex-M1-preview (2026) from the Xiaomi_Robotics_1 adapter in XPolicyLab.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${REX_M1_PREVIEW_CONDA_ENV:-rex_m1_preview}"

echo "[Rex_M1_preview] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"
echo "[Rex_M1_preview] CONDA_ENV=${CONDA_ENV}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please install Miniconda/Anaconda first." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.12 -y
fi
conda activate "${CONDA_ENV}"

# XPolicyLab itself: client_server.ws imports websockets, msgpack and
# msgpack_numpy, and the vendored requirements bring in none of them, so without
# this the server dies before it binds the port.
python -m pip install -e "${XPOLICYLAB_ROOT}"

# Core dependencies
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1 scipy numpy Pillow ninja psutil
# ninja and psutil are imported by flash-attn's setup.py. --no-build-isolation
# tells pip not to fetch build requirements, so they must already be installed.
pip install flash-attn==2.8.3 --no-build-isolation

# Vendored xr1 requirements. mmengine (model registry) and liger-kernel (fused
# RMSNorm/RoPE in the VLM) are needed just to import mibot, so inference needs
# them too; the rest are pulled in by the training entrypoints.
pip install -r "${POLICY_DIR}/rex_m1/assets/requirements.txt"

# Image/video IO used by XPolicyLab's shared data utilities. Not part of the
# vendored requirements, which cover the model itself only.
pip install opencv-python-headless h5py imageio imageio-ffmpeg tqdm

echo "[Rex_M1_preview] Installation finished."
echo "[Rex_M1_preview] Activate env: conda activate ${CONDA_ENV}"
