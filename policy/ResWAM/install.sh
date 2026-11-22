#!/bin/bash
set -euo pipefail

# Usage: bash install.sh [venv_dir]      (default: policy/ResWAM/.venv)
# Clones ResWAM at the commit behind the released checkpoints into policy/ResWAM/ResWAM and
# builds its Python 3.10 environment (PyTorch 2.6.0 / CUDA 12.4, transformers 5.2.0) with uv,
# then installs XPolicyLab into the same environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/ResWAM"
RESWAM_COMMIT=c8564aa33ca1a039cbe7b1ec2ac27d203adb136f
VENV_DIR="$(realpath -m "${1:-${SCRIPT_DIR}/.venv}")"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    git clone https://github.com/OPTML-Group/ResWAM.git "${UPSTREAM_DIR}"
fi
git -C "${UPSTREAM_DIR}" fetch --quiet origin
git -C "${UPSTREAM_DIR}" checkout --quiet "${RESWAM_COMMIT}"

bash "${UPSTREAM_DIR}/scripts/env/setup_reswam.sh" --venv "${VENV_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" -e "${XPL_ROOT}"

echo "[install] policy env: ${VENV_DIR}  (pass it as <policy_conda_env> to eval.sh)"
