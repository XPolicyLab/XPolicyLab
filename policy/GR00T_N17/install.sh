# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
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
# NOTE: pyproject pins `[tool.uv] required-environments` to both x86_64 and aarch64,
# and the aarch64 torchcodec/flash-attn wheels under scripts/deployment/dgpu/wheels/
# are not shipped. `uv sync` therefore fails to lock on an x86_64 GPU host because it
# must resolve the (missing) aarch64 wheels. We instead create the venv and use
# `uv pip install -e .`, which resolves only for the *current* platform (x86_64) and
# honors [tool.uv.sources] / [[tool.uv.index]] while ignoring required-environments.
uv venv --clear --python 3.10
uv pip install -e .
uv run python -c "import gr00t; print('GR00T ok')"

uv pip install -e "${XPOLICYLAB_ROOT}"
uv pip install h5py pyyaml
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[GR00T_N17] Installation finished."
echo "[GR00T_N17] Policy server env: source ${GR00T_ROOT}/.venv/bin/activate"
echo "[GR00T_N17] Eval example:"
echo "  bash eval.sh RoboDojo sweep_blocks cotrain arx_x5 3500 joint 0 0 0 uv mibot"
