#!/bin/bash
set -euo pipefail
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
policy_conda_env="${1:-dreamzero_robodojo}"
python_version="${DREAMZERO_PYTHON_VERSION:-3.11}"

echo "[DreamZero install] Preparing conda env: ${policy_conda_env}"
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "${conda_exe}" ]; then
    echo "[DreamZero install][ERROR] conda executable not found. Activate conda or set CONDA_EXE." >&2
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

echo "[DreamZero install] Installing DreamZero package in editable mode."
cd "${SCRIPT_DIR}/dreamzero"
python -m pip install -e . --extra-index-url "${DREAMZERO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"

echo "[DreamZero install] Restoring DreamZero-compatible pinned packages."
python -m pip install --force-reinstall \
    "numpy==1.26.4" \
    "opencv-python==4.8.0.74" \
    "datasets==3.6.0" \
    "rerun-sdk==0.21.0" \
    "huggingface-hub>=0.34.2,<0.36.0" \
    "packaging<26.0,>=24.2"

if [ "${INSTALL_LEROBOT:-0}" = "1" ]; then
    echo "[DreamZero install] Installing LeRobot >= 0.4.0 for optional data utilities."
    python -m pip install --no-deps --upgrade "lerobot>=0.4.0"
else
    echo "[DreamZero install] Skipping LeRobot package. Direct v3 training uses DreamZero's loader."
fi

if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    echo "[DreamZero install] Installing flash-attn."
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install --no-build-isolation flash-attn
else
    echo "[DreamZero install] Skipping flash-attn. Set INSTALL_FLASH_ATTN=1 to install it."
fi

if [ "${INSTALL_TRANSFORMER_ENGINE:-0}" = "1" ]; then
    echo "[DreamZero install] Installing transformer_engine."
    python -m pip install --no-build-isolation "transformer_engine[pytorch]"
else
    echo "[DreamZero install] Skipping transformer_engine. Set INSTALL_TRANSFORMER_ENGINE=1 to install it."
fi

echo "[DreamZero install] Installing XPolicyLab package in editable mode."
cd "${ROOT_DIR}/XPolicyLab"
python -m pip install -e .

python - <<'PY'
import importlib.metadata as metadata

for package in ("dreamzero", "lerobot"):
    try:
        print(f"[DreamZero install] {package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"[DreamZero install][WARN] {package} is not installed")
PY

echo "[DreamZero install] Done. Activate with: conda activate ${policy_conda_env}"
