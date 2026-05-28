#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOVLA_ROOT="${POLICY_DIR}/smovla"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
VENV_DIR="${POLICY_DIR}/.venv"

echo "[SmolVLA] POLICY_DIR=${POLICY_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3.10 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

cd "${SMOVLA_ROOT}"
pip install -e ".[smolvla]"

cd "${XPOLICYLAB_ROOT}"
pip install -e .
pip install h5py

python -c "import lerobot; from lerobot.policies.factory import get_policy_class; print('smolvla:', get_policy_class('smolvla'))"
python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[SmolVLA] Installation finished."
echo "[SmolVLA] Activate: source ${VENV_DIR}/bin/activate"
