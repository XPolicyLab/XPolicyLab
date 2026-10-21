#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${1:-phoenix-mwa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "${SCRIPT_DIR}/requirements-joint96.txt"
python -m pip install -e "${XPL_ROOT}"
echo "Installed ${ENV_NAME}. Run download_joint96_weights.sh; source/config/tokenizer are checked into joint96_source/."
