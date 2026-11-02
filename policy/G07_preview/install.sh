#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-g07_preview}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
# The vendored inference code (src/) runs from PYTHONPATH; these are its third-party imports.
pip install numpy==2.2.6 pillow==12.0.0 omegaconf==2.3.0 hydra-core==1.3.2 safetensors==0.5.3 \
    transformers==4.49.0 tqdm==4.68.3 jaxtyping==0.3.7 einops==0.8.1 typing_extensions==4.15.0 \
    rich==14.2.0 imageio==2.37.0 accelerate==1.12.0 huggingface-hub==0.29.2
pip install -e "${XPL_ROOT}"
