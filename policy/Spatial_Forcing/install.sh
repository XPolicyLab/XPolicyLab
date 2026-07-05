# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SF_ROOT="${POLICY_DIR}/open_sf"
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
    echo "[Spatial_Forcing][ERROR] XPolicyLab root not found above ${1}" >&2
    return 1
}
XPOLICYLAB_ROOT="$(find_xpolicylab_root "${POLICY_DIR}")"

echo "[Spatial_Forcing] OPEN_SF_ROOT=${OPEN_SF_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

TJY_ROOT="$(cd "${POLICY_DIR}/../../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TJY_ROOT}/.cache/uv}"
mkdir -p "${UV_CACHE_DIR}"
echo "[Spatial_Forcing] UV_CACHE_DIR=${UV_CACHE_DIR}"

cd "${OPEN_SF_ROOT}"
rm -rf .venv
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install -e "${XPOLICYLAB_ROOT}"
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"

chmod +x "${OPEN_SF_ROOT}/.venv/bin/python"* 2>/dev/null || true
if ! "${OPEN_SF_ROOT}/.venv/bin/python" -c "import jax; print('jax ok')" 2>/dev/null; then
  echo "[Spatial_Forcing][ERROR] jax not available in ${OPEN_SF_ROOT}/.venv" >&2
  exit 1
fi

echo "[Spatial_Forcing] Installation finished."
echo "[Spatial_Forcing] Activate: source ${OPEN_SF_ROOT}/.venv/bin/activate"
