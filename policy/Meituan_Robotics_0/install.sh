#!/bin/bash
set -euo pipefail

# Run inside a fresh Python 3.11 environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# XPolicyLab depends on opencv-python-headless>=4.8. Pin the NumPy-1-compatible
# OpenCV line before installing XPolicyLab so pip does not select OpenCV 5.
python -m pip install "numpy<2" "opencv-python-headless>=4.10,<5"
python -m pip install -e "${SCRIPT_DIR}/../.."
python -m pip install "torch==2.7.1" "torchvision==0.22.1" "transformers==5.7.0" "omegaconf>=2.3" "pillow>=10" "numpy<2" "tqdm>=4.66" "pyyaml>=6" "huggingface_hub>=0.34"
python -m pip install "flash-attn==2.8.3" "flash-linear-attention==0.5.1" "causal-conv1d==1.6.2.post1" --no-build-isolation
