#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
policy_conda_env=${1:-${OLA_SEM_CONDA_ENV:-ola_sem}}
python_version=${OLA_SEM_PYTHON_VERSION:-3.10}
pip_index_url=${OLA_SEM_PIP_INDEX_URL:-https://pypi.org/simple}
torch_index_url=${OLA_SEM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
torch_version=${OLA_SEM_TORCH_VERSION:-2.7.1}
torchvision_version=${OLA_SEM_TORCHVISION_VERSION:-0.22.1}
flash_attn_version=${OLA_SEM_FLASH_ATTN_VERSION:-2.8.3}

usage() {
    cat <<'EOF'
Usage: bash install.sh [conda_env_name_or_prefix]

Creates a Python 3.10 conda environment when needed and installs the complete
OLA-SEM policy runtime/training stack. The default environment name is ola_sem.

Useful overrides:
  OLA_SEM_CONDA_ENV                  Default conda environment name
  OLA_SEM_PYTHON_VERSION             Python version (default: 3.10)
  OLA_SEM_PIP_INDEX_URL              Python package index
  OLA_SEM_TORCH_INDEX_URL            PyTorch wheel index (default: cu128)
  OLA_SEM_TORCH_VERSION              PyTorch version (default: 2.7.1)
  OLA_SEM_TORCHVISION_VERSION        torchvision version (default: 0.22.1)
  OLA_SEM_SKIP_TORCH_INSTALL=1       Keep PyTorch already present in the env
  OLA_SEM_FLASH_ATTN_VERSION         flash-attn version (default: 2.8.3)
  OLA_SEM_FLASH_ATTN_WHEEL           Install a compatible local wheel/URL
  OLA_SEM_SKIP_FLASH_ATTN_INSTALL=1  Skip FlashAttention installation
  OLA_SEM_MAX_JOBS                   Parallel jobs for a source build (default: 4)
EOF
}

if [[ "${policy_conda_env}" == "-h" || "${policy_conda_env}" == "--help" ]]; then
    usage
    exit 0
fi

conda_exe=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "${conda_exe}" ]]; then
    echo "[OLA_SEM install][ERROR] conda was not found. Install Miniconda/Miniforge first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$("${conda_exe}" info --base)/etc/profile.d/conda.sh"

if ! conda activate "${policy_conda_env}" >/dev/null 2>&1; then
    echo "[OLA_SEM install] Creating conda environment: ${policy_conda_env}"
    if [[ "${policy_conda_env}" == */* ]]; then
        conda create -y -p "${policy_conda_env}" "python=${python_version}"
    else
        conda create -y -n "${policy_conda_env}" "python=${python_version}"
    fi
    conda activate "${policy_conda_env}"
else
    echo "[OLA_SEM install] Reusing conda environment: ${policy_conda_env}"
fi

python -m pip install --index-url "${pip_index_url}" --upgrade \
    pip setuptools wheel packaging ninja

if [[ "${OLA_SEM_SKIP_TORCH_INSTALL:-0}" != "1" ]]; then
    echo "[OLA_SEM install] Installing torch==${torch_version}, torchvision==${torchvision_version}"
    python -m pip install \
        "torch==${torch_version}" \
        "torchvision==${torchvision_version}" \
        --index-url "${torch_index_url}"
else
    echo "[OLA_SEM install] Keeping the environment's existing PyTorch installation."
fi

echo "[OLA_SEM install] Installing OLA-SEM runtime and training dependencies."
python -m pip install --index-url "${pip_index_url}" \
    -r "${SCRIPT_DIR}/ola_sem/requirements.txt"

if [[ "${OLA_SEM_SKIP_FLASH_ATTN_INSTALL:-0}" != "1" ]]; then
    if [[ -n "${OLA_SEM_FLASH_ATTN_WHEEL:-}" ]]; then
        echo "[OLA_SEM install] Installing FlashAttention wheel: ${OLA_SEM_FLASH_ATTN_WHEEL}"
        python -m pip install "${OLA_SEM_FLASH_ATTN_WHEEL}"
    else
        echo "[OLA_SEM install] Building flash-attn==${flash_attn_version}; CUDA toolkit/nvcc is required."
        MAX_JOBS="${OLA_SEM_MAX_JOBS:-4}" \
            python -m pip install --index-url "${pip_index_url}" \
                --no-build-isolation "flash-attn==${flash_attn_version}"
    fi
else
    echo "[OLA_SEM install] Skipping FlashAttention installation as requested."
fi

python -m pip install --index-url "${pip_index_url}" \
    "websockets>=14.0" "msgpack-numpy>=0.4.8"
python -m pip install -e "${XPL_ROOT}" --no-build-isolation --no-deps

python - <<'PY'
import importlib.metadata as metadata
import os
import sys

import torch
import transformers
import XPolicyLab

if os.environ.get("OLA_SEM_SKIP_FLASH_ATTN_INSTALL", "0") != "1":
    import flash_attn

required = (
    "torch",
    "torchvision",
    "transformers",
    "deepspeed",
    "diffusers",
    "qwen-vl-utils",
    "websockets",
    "msgpack-numpy",
)
for package in required:
    print(f"[OLA_SEM install] {package}=={metadata.version(package)}")
try:
    print(f"[OLA_SEM install] flash-attn=={metadata.version('flash-attn')}")
except metadata.PackageNotFoundError:
    print("[OLA_SEM install][WARN] flash-attn is not installed; OLA-SEM inference will not run.")
print(f"[OLA_SEM install] python={sys.executable}")
print(f"[OLA_SEM install] XPolicyLab={XPolicyLab.__file__}")
PY

echo "[OLA_SEM install] Done. Activate with: conda activate ${policy_conda_env}"
