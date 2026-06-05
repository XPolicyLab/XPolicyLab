# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIRIT_ROOT="${POLICY_DIR}/spirit_v15"
SPIRIT_VENV="${SPIRIT_ROOT}/.venv"
SPIRIT_PYTHON="${SPIRIT_VENV}/bin/python"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[Spirit_v15] SPIRIT_ROOT=${SPIRIT_ROOT}"
echo "[Spirit_v15] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"

# NFS/cache on different filesystems: avoid hardlink warnings from uv.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if command -v uv >/dev/null 2>&1; then
  cd "${SPIRIT_ROOT}"
  uv sync --extra train
  uv pip install -e . --python "${SPIRIT_PYTHON}"
  cd "${XPOLICYLAB_ROOT}"
  uv pip install -e . --python "${SPIRIT_PYTHON}"
  "${SPIRIT_PYTHON}" -c "import XPolicyLab; print('XPolicyLab ok')" 2>/dev/null || true
else
  echo "[Spirit_v15] uv not found, using pip/venv fallback"
  cd "${SPIRIT_ROOT}"
  python -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements-base.txt
  pip install -r requirements-train.txt
  pip install -e .
  cd "${XPOLICYLAB_ROOT}"
  pip install -e .
fi

echo "[Spirit_v15] Installation finished."
echo "[Spirit_v15] Activate: source ${SPIRIT_ROOT}/.venv/bin/activate"
echo "[Spirit_v15] Pretrained: export SPIRIT_PRETRAINED_PATH=<hf_repo_or_local_dir>"
