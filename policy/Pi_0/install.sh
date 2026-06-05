# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${POLICY_DIR}/openpi"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[Pi_0] OPENPI_ROOT=${OPENPI_ROOT}"
echo "[Pi_0] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${OPENPI_ROOT}"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run python -c "import openpi; print('openpi ok')" 2>/dev/null || uv run python -c "print('openpi env ok')"

uv pip install -e "${XPOLICYLAB_ROOT}"
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[Pi_0] Installation finished."
echo "[Pi_0] Activate: source ${OPENPI_ROOT}/.venv/bin/activate"
echo "[Pi_0] Train: bash ${POLICY_DIR}/train.sh ..."
