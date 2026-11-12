#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Dependencies are installed only when the user runs this script.
python3.11 -m venv "${POLICY_VENV:-${DIR}/.venv}"
PY="${POLICY_VENV:-${DIR}/.venv}/bin/python"
export PATH="$(dirname "${PY}"):${PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
"${PY}" -m pip install --upgrade pip 'setuptools<82' wheel packaging ninja psutil
"${PY}" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple
"${PY}" -m pip install -r "${DIR}/requirements.txt"
# Source-built extension wheel tags do not encode the PyTorch ABI.
"${PY}" -m pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3 causal-conv1d==1.6.2.post1
"${PY}" -m pip check
"${PY}" -c 'import torch, flash_attn, causal_conv1d; print("Era0 policy environment installed successfully")'
