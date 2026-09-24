#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICY_ENV="${1-sipai-eval}"

if [[ "${POLICY_ENV}" == "-h" || "${POLICY_ENV}" == "--help" ]]; then
    echo "Usage: bash install.sh [conda_env_name_or_prefix]"
    echo "Create or reuse a Python 3.11 environment (default: sipai-eval)."
    echo "Reuse installed CUDA PyTorch 2.6-2.8; otherwise install 2.7.0 CUDA 12.8."
    echo "Install SIPAI inference dependencies and XPolicyLab."
    echo "Requires Conda and a compatible NVIDIA driver; RoboDojo simulator is separate."
    exit 0
fi
if [[ $# -gt 1 || -z "${POLICY_ENV}" ]]; then
    echo "Usage: bash install.sh [conda_env_name_or_prefix]" >&2
    exit 2
fi

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
    echo "Conda not found. Install Miniconda/Miniforge or set CONDA_EXE." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$("${CONDA_BIN}" info --base)/etc/profile.d/conda.sh"

if ! conda activate "${POLICY_ENV}" >/dev/null 2>&1; then
    if [[ "${POLICY_ENV}" == */* ]]; then
        conda create -y -p "${POLICY_ENV}" python=3.11 pip
    else
        conda create -y -n "${POLICY_ENV}" python=3.11 pip
    fi
    conda activate "${POLICY_ENV}"
fi
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("SIPAI requires Python 3.11; select a new environment name or path.")
print(f"[SIPAI] Installing into {sys.prefix}", flush=True)
PY

# Reuse installed packages and pip's configured cache; do not force upgrades.
python -m pip install 'setuptools>=61.0' wheel
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") is not None else 1)
PY
then
    echo "[SIPAI] Reusing installed PyTorch."
else
    python -m pip install 'torch==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128
fi
python -m pip install \
    'numpy>=1.26,<2' \
    'sentencepiece==0.2.2' \
    'safetensors>=0.4.3' \
    'huggingface_hub>=0.34.0' \
    'pillow' \
    'opencv-python-headless==4.11.0.86' \
    --no-build-isolation -e "${XPL_ROOT}"
python -m pip check

PYTHONPATH="${XPL_ROOT}:${XPL_ROOT}/..${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
import torch
from XPolicyLab.policy.SIPAI.model import Model
from XPolicyLab.client_server.ws.model_server import PolicyServer

torch_series = tuple(int(part) for part in torch.__version__.split(".")[:2])
if torch_series not in {(2, 6), (2, 7), (2, 8)} or torch.version.cuda is None:
    raise SystemExit("Use a CUDA build of PyTorch 2.6, 2.7 or 2.8 in the selected environment.")
print(f"[SIPAI] PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
print("[SIPAI] Model and WebSocket server imports passed.")
PY
echo "[SIPAI] Installation complete. Activate with: conda activate ${POLICY_ENV}"
