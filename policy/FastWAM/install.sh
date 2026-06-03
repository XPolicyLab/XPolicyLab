#!/bin/bash
set -euo pipefail

ENV_NAME=${1:-fastwam}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/FastWAM"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

conda install -c conda-forge "ffmpeg>=6,<7" -y
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install opencv-python
pip install -e "${FASTWAM_DIR}"
pip install -e "${ROOT_DIR}/XPolicyLab"
