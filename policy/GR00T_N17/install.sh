#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[GR00T_N17] GR00T_ROOT=${GR00T_ROOT}"
echo "[GR00T_N17] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${GR00T_ROOT}"
uv sync --python 3.10
uv run python -c "import gr00t; print('GR00T ok')"

uv pip install -e "${XPOLICYLAB_ROOT}"
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[GR00T_N17] Installation finished."
echo "[GR00T_N17] Policy server env: source ${GR00T_ROOT}/.venv/bin/activate"
echo "[GR00T_N17] Eval example:"
echo "  bash eval.sh RoboDojo sweep_blocks cotrain arx_x5 3500 joint 0 0 0 uv mibot"
