#!/usr/bin/env bash
set -euo pipefail

# Policy environment for PICA_WAM evaluation and training.
# Usage: bash install.sh [env_name]
ENV_NAME="${1:-pica-wam}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.11 -y
conda activate "${ENV_NAME}"

# Triton compiles a small CUDA driver helper on first use, and torchcodec links
# the FFmpeg shared libraries; install both so the host image needs neither.
conda install -c conda-forge -y gcc_linux-64 gxx_linux-64 "ffmpeg>=6,<8"

python -m pip install --upgrade pip
python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
# The vendored InternW0-Delta + PICA-WAM source, with DeepSpeed for training.
# The PICA-WAM checkpoint was trained and evaluated without the optional
# flash-linear-attention / causal-conv1d kernels, so they are left out here.
python -m pip install -e "${SCRIPT_DIR}/internw0_pica[train,modelscope]" "ftfy==6.3.1"
# Keep the numpy pinned by internw0_pica; an unconstrained resolve pulls an
# OpenCV release that requires numpy 2.
python -m pip install -e "${XPL_ROOT}" "numpy==1.26.4"

echo "Installed policy environment: ${ENV_NAME}"
echo "Next: conda activate ${ENV_NAME} && bash ${SCRIPT_DIR}/download_assets.sh"
