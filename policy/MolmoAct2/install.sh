#!/usr/bin/env bash
# Install the direct-Hugging-Face MolmoAct2 inference environment.

set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
VENV_DIR="${POLICY_DIR}/.venv"
PYTHON_VERSION="${MOLMOACT2_PYTHON:-3.11}"
USE_SYSTEM_TORCH="${MOLMOACT2_USE_SYSTEM_TORCH:-0}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[MolmoAct2] Creating ${VENV_DIR} with Python ${PYTHON_VERSION}"
    if [[ "${USE_SYSTEM_TORCH}" == "1" ]]; then
        uv venv --system-site-packages --python "${PYTHON_VERSION}" "${VENV_DIR}"
    else
        uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    fi
else
    echo "[MolmoAct2] Reusing ${VENV_DIR}"
fi

if [[ "${USE_SYSTEM_TORCH}" == "1" ]]; then
    UV_LINK_MODE=copy uv pip install \
        --python "${VENV_DIR}/bin/python" \
        --no-deps "torchvision==0.24.0"
    UV_LINK_MODE=copy uv pip install \
        --python "${VENV_DIR}/bin/python" \
        "einops==0.8.1" \
        "transformers==5.14.1" \
        "huggingface-hub>=0.36" \
        "safetensors>=0.5" \
        "pillow>=10" \
        "sympy>=1.13.3,<2" \
        "networkx>=2.5.1" \
        "jinja2>=3.1" \
        "scipy==1.15.3" \
        "requests>=2.32" \
        "numpy>=1.26"
else
    UV_LINK_MODE=copy uv pip install \
        --python "${VENV_DIR}/bin/python" \
        "torch>=2.7,<3" \
        "torchvision==0.24.0" \
        "einops==0.8.1" \
        "transformers==5.14.1" \
        "huggingface-hub>=0.36" \
        "safetensors>=0.5" \
        "pillow>=10" \
        "requests>=2.32" \
        "numpy>=1.26"
fi

UV_LINK_MODE=copy uv pip install \
    --python "${VENV_DIR}/bin/python" \
    -e "${XPOLICYLAB_ROOT}"

"${VENV_DIR}/bin/python" - <<'PY'
import einops
import requests
import torch
import torchvision
import transformers
import XPolicyLab

print("[MolmoAct2] XPolicyLab import: PASS")
print(f"[MolmoAct2] torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"[MolmoAct2] torchvision={torchvision.__version__} einops={einops.__version__}")
print(f"[MolmoAct2] transformers={transformers.__version__}")
PY

echo "[MolmoAct2] Installation finished"
echo "[MolmoAct2] Activate with: source ${VENV_DIR}/bin/activate"
