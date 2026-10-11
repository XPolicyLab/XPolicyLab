#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy_env="${1:-me_u0}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${policy_env}" python=3.12 -y
conda activate "${policy_env}"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
python -m pip install flash-attn==2.8.3 --no-build-isolation
