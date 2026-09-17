# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPSILONVLA_ROOT="${POLICY_DIR}/epsilonvla"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[EpsilonVLA] EPSILONVLA_ROOT=${EPSILONVLA_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${EPSILONVLA_ROOT}"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install -e "${XPOLICYLAB_ROOT}"
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[EpsilonVLA] Installation finished."
echo "[EpsilonVLA] Activate: source ${EPSILONVLA_ROOT}/.venv/bin/activate"
