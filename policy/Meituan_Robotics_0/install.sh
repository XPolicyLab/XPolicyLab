#!/usr/bin/env bash
set -euo pipefail

# Run in a dedicated Python 3.11 policy environment, inside official XPolicyLab.
# Source builds of CUDA extensions need a compatible CUDA 12.x toolkit and C++ compiler.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import platform, sys; assert sys.version_info[:2] == (3, 11) and platform.system() == "Linux" and platform.machine() == "x86_64", "Use Python 3.11 on Linux x86_64"'
if [[ ! -f "$SCRIPT_DIR/../../pyproject.toml" ]]; then
    echo "Place this policy under the official XPolicyLab/policy/ directory first." >&2
    exit 1
fi
"$PYTHON" -m pip install \
    "setuptools==68.2.2" "wheel==0.41.2" "packaging==26.2" "ninja==1.13.0" "psutil==7.2.2"
"$PYTHON" -m pip install \
    "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" \
    --index-url https://download.pytorch.org/whl/cu126

# Pin preprocessing, model kernels, weight loading, and the shared RPC transport.
# safetensors 0.8.0rc0 is the public prerelease used by the tested environment.
"$PYTHON" -m pip install -e "$SCRIPT_DIR/../.." \
    "transformers==5.7.0" "tokenizers==0.22.2" "safetensors==0.8.0rc0" \
    "huggingface_hub==1.15.0" "triton==3.3.1" \
    "fla-core==0.5.1" "flash-linear-attention==0.5.1" \
    "numpy==2.4.6" "opencv-python-headless==4.13.0.92" "pillow==12.2.0" \
    "omegaconf==2.3.0" "einops==0.8.2" "pyyaml==6.0.3" \
    "h5py==3.16.0" "websockets==16.0" "msgpack==1.1.2" \
    "msgpack-numpy==0.4.8" "pydantic==2.13.4"
MAX_JOBS="${MAX_JOBS:-4}" "$PYTHON" -m pip install \
    "flash-attn==2.8.3" "causal-conv1d==1.6.2.post1" --no-build-isolation
"$PYTHON" -m pip check
"$PYTHON" -c 'import torch, torchvision; assert torch.cuda.is_available(), "The policy import check requires a visible CUDA GPU"; import flash_attn, fla, causal_conv1d; from transformers import Qwen3_5ForConditionalGeneration; print("Policy dependencies import successfully; model weights have not been loaded.")'
