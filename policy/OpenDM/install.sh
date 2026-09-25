#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${OPENDM_CONDA_ENV:-opendm}"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10 -y
fi
conda activate "${CONDA_ENV}"
python -m pip install --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "${POLICY_DIR}/requirements.lock"
if [[ "${OPENDM_INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
    MAX_JOBS="${MAX_JOBS:-2}" python -m pip install \
        --index-url https://pypi.org/simple --no-build-isolation --no-deps flash-attn==2.8.3.post1
fi
# Import through .pth files without generating egg-info inside the vendor tree.
OPENDM_POLICY_DIR="${POLICY_DIR}" python - <<'PY'
import os
from pathlib import Path
import site
root = Path(os.environ["OPENDM_POLICY_DIR"])
Path(site.getsitepackages()[0], "xpolicylab_opendm.pth").write_text(
    str(root / "opendm") + "\n" + str(root.parents[2]) + "\n"
)
PY
export PYTHONDONTWRITEBYTECODE=1
python -c "import opendm, XPolicyLab, torch; print('OpenDM ready; CUDA:', torch.cuda.is_available())"
python -m pip check
