#!/bin/bash
# Builds the policy environment in openpi/.venv with uv and installs XPolicyLab into it.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "[SLA_Memory][ERROR] uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

cd "${POLICY_DIR}/openpi"
uv sync --frozen
uv pip install --python .venv/bin/python -e "${XPL_ROOT}"
PYTHONPATH="$(cd "${XPL_ROOT}/.." && pwd)" .venv/bin/python -c \
    "import jax, openpi, XPolicyLab.utils.process_data; print('[SLA_Memory] jax', jax.__version__, jax.devices())"
echo "[SLA_Memory] Installation finished: ${POLICY_DIR}/openpi/.venv"
