#!/bin/bash
set -euo pipefail

# Mem_0 policy environment setup (Conda). Docker is not used.
# Run from this policy folder: bash install.sh [policy_conda_env]
# Default env name: mem0. The planning module uses a separate env (see bottom).

policy_conda_env="${1:-mem0}"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${POLICY_DIR}/Mem_0"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Execution module + data + inference env (Pytorch 2.6 + CUDA 12.4) ---
conda create -n "${policy_conda_env}" python=3.10 -y
conda activate "${policy_conda_env}"

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torchcodec --index-url https://download.pytorch.org/whl/cu124
pip install -r "${UPSTREAM_DIR}/requirements.txt"
pip install "flash-attn==2.6.1" --no-build-isolation
conda install "ffmpeg" -c conda-forge -y
pip install openai opencv-python 

# XPolicyLab package (needed by process_data.sh / model.py imports).
pip install -e "${ROOT_DIR}/XPolicyLab"

echo "[install] ${policy_conda_env} ready."
echo "[install] Download backbones:  cd ${UPSTREAM_DIR}/checkpoints && python _download.py"
echo "[install] Planning module (Mn): bash install_planning.sh — see INSTALLATION.md."
