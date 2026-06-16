#!/usr/bin/env bash
# XPolicyLab deploy: policy server env=gigaworld-policy; run setup_eval_policy_server.sh with this env.
set -euo pipefail
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
policy_conda_env="${1:-${GIGAWORLD_CONDA_ENV:-gigaworld-policy}}"
python_version="${GIGAWORLD_PYTHON_VERSION:-3.11}"

echo "[GigaWorldPolicy install] Preparing conda env: ${policy_conda_env}"
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "${conda_exe}" ]; then
    echo "[GigaWorldPolicy install][ERROR] conda executable not found. Activate conda or set CONDA_EXE." >&2
    exit 1
fi
source "$("${conda_exe}" info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${policy_conda_env}"; then
    conda create -y -n "${policy_conda_env}" "python=${python_version}"
fi
conda activate "${policy_conda_env}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first with a CUDA-matched build so the rest of the deps resolve
# against it. The validated env uses cu128 (torch 2.8.0 / torchvision 0.23.0).
# Override TORCH_INDEX_URL / TORCH_VERSION / TORCHVISION_VERSION as needed, or set
# SKIP_TORCH_INSTALL=1 if a suitable torch is already installed.
if [ "${SKIP_TORCH_INSTALL:-0}" != "1" ]; then
    torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    torch_version="${TORCH_VERSION:-2.8.0}"
    torchvision_version="${TORCHVISION_VERSION:-0.23.0}"
    echo "[GigaWorldPolicy install] Installing torch==${torch_version} torchvision==${torchvision_version} from ${torch_index_url}"
    python -m pip install \
        "torch==${torch_version}" \
        "torchvision==${torchvision_version}" \
        --index-url "${torch_index_url}"
fi

echo "[GigaWorldPolicy install] Installing XPolicyLab package in editable mode."
cd "${XPOLICYLAB_ROOT}"
python -m pip install -e .

echo "[GigaWorldPolicy install] Installing GigaWorldPolicy package in editable mode."
cd "${SCRIPT_DIR}/giga_world_policy"
python -m pip install -e .

python - <<'PYVERIFY'
import importlib.metadata as metadata
import sys

import XPolicyLab
import world_action_model

print(f"[GigaWorldPolicy install] python={sys.executable}")
print(f"[GigaWorldPolicy install] xpolicylab=={metadata.version('xpolicylab')}")
print(f"[GigaWorldPolicy install] gwp-xpl=={metadata.version('gwp-xpl')}")
print(f"[GigaWorldPolicy install] XPolicyLab={XPolicyLab.__file__}")
print(f"[GigaWorldPolicy install] world_action_model={world_action_model.__file__}")
PYVERIFY

echo "[GigaWorldPolicy install] Done. Activate with: conda activate ${policy_conda_env}"
