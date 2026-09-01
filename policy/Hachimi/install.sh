#!/usr/bin/env bash
# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${POLICY_DIR}/open_sf_wam"
find_xpolicylab_root() {
    local dir
    dir="$(cd "${1}" && pwd)"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/setup_policy_server.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo "[Hachimi][ERROR] XPolicyLab root not found above ${1}" >&2
    return 1
}
XPOLICYLAB_ROOT="$(find_xpolicylab_root "${POLICY_DIR}")"

echo "[Hachimi] CODE_ROOT=${CODE_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

WORKSPACE_ROOT="$(cd "${POLICY_DIR}/../../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/uv}"
mkdir -p "${UV_CACHE_DIR}"
echo "[Hachimi] UV_CACHE_DIR=${UV_CACHE_DIR}"

cd "${CODE_ROOT}"
rm -rf .venv
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install -e "${XPOLICYLAB_ROOT}"
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"

chmod +x "${CODE_ROOT}/.venv/bin/python"* 2>/dev/null || true
if ! "${CODE_ROOT}/.venv/bin/python" -c "import jax; print('jax ok')" 2>/dev/null; then
  echo "[Hachimi][ERROR] jax not available in ${CODE_ROOT}/.venv" >&2
  exit 1
fi

"${CODE_ROOT}/.venv/bin/python" - <<'PY'
import openpi.training.config as c
assert "pi05sfwam_jax_robodojo_v21_merged" in c._CONFIGS_DICT, "Hachimi train config not registered"
print("Hachimi train config registered ok")
PY

echo "[Hachimi] Installation finished."
echo "[Hachimi] Activate: source ${CODE_ROOT}/.venv/bin/activate"
