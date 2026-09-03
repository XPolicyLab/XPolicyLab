#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENDM_ROOT="${POLICY_DIR}/opendm"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${OPENDM_CONDA_ENV:-opendm}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        conda create -n "${CONDA_ENV}" python=3.10 -y
    fi
    conda activate "${CONDA_ENV}"
fi

python -m pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url "${OPENDM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install ninja packaging

if [[ "${OPENDM_INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    MAX_JOBS="${MAX_JOBS:-2}" python -m pip install flash-attn --no-build-isolation
fi

python -m pip install -e "${OPENDM_ROOT}"
python -m pip install orjson
python -m pip install -e "${XPOLICYLAB_ROOT}"

python -c "import opendm; print('OpenDM import OK')"
python -c "import XPolicyLab; print('XPolicyLab import OK')"
echo "[OpenDM] installation complete in ${CONDA_ENV}"
