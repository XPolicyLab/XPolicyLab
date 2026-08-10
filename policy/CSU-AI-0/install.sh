#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SF_ROOT="${POLICY_DIR}/open_sf"
RUNTIME_REPO="${CSU_AI_0_RUNTIME_REPO:-https://github.com/XPolicyLab/XPolicyLab.git}"
RUNTIME_REVISION="${CSU_AI_0_RUNTIME_REVISION:-2dfd4ee7af7ecf8b3281847179f14d22c5c04a35}"
RUNTIME_SUBDIR="policy/Spatial_Forcing/open_sf"

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
    echo "[CSU-AI-0][ERROR] XPolicyLab root not found above ${1}" >&2
    return 1
}

XPOLICYLAB_ROOT="$(find_xpolicylab_root "${POLICY_DIR}")"

if ! command -v git >/dev/null 2>&1; then
    echo "[CSU-AI-0][ERROR] git is required to install the pinned OpenPI runtime source" >&2
    exit 1
fi
UV_BIN="${CSU_AI_0_UV_BIN:-}"
if [[ -z "${UV_BIN}" ]]; then
    UV_BIN="$(command -v uv 2>/dev/null || true)"
fi
if [[ -z "${UV_BIN}" ]]; then
    for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv" /usr/local/bin/uv /usr/bin/uv; do
        if [[ -x "${candidate}" ]]; then
            UV_BIN="${candidate}"
            break
        fi
    done
fi
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
    echo "[CSU-AI-0][ERROR] uv is required; install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

install_runtime_source() {
    local stage_dir
    local clone_dir
    stage_dir="$(mktemp -d "${POLICY_DIR}/.open_sf.install.XXXXXX")"
    clone_dir="${stage_dir}/XPolicyLab"

    cleanup_runtime_stage() {
        if [[ -n "${stage_dir:-}" && -d "${stage_dir}" && "${stage_dir}" == "${POLICY_DIR}"/.open_sf.install.* ]]; then
            rm -rf -- "${stage_dir}"
        fi
    }
    trap cleanup_runtime_stage EXIT INT TERM

    echo "[CSU-AI-0] Fetching pinned runtime source ${RUNTIME_REVISION}"
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "${RUNTIME_REPO}" "${clone_dir}"
    git -C "${clone_dir}" sparse-checkout init --cone
    git -C "${clone_dir}" sparse-checkout set "${RUNTIME_SUBDIR}"
    GIT_LFS_SKIP_SMUDGE=1 git -C "${clone_dir}" checkout --detach "${RUNTIME_REVISION}"

    local fetched_runtime="${clone_dir}/${RUNTIME_SUBDIR}"
    if [[ ! -f "${fetched_runtime}/pyproject.toml" || ! -f "${fetched_runtime}/uv.lock" ]]; then
        echo "[CSU-AI-0][ERROR] pinned runtime source is incomplete: ${fetched_runtime}" >&2
        exit 1
    fi
    if [[ -e "${OPEN_SF_ROOT}" ]]; then
        echo "[CSU-AI-0][ERROR] refusing to overwrite existing incomplete runtime: ${OPEN_SF_ROOT}" >&2
        exit 1
    fi

    mv "${fetched_runtime}" "${OPEN_SF_ROOT}"
    {
        echo "repository=${RUNTIME_REPO}"
        echo "revision=${RUNTIME_REVISION}"
        echo "source_subdir=${RUNTIME_SUBDIR}"
    } > "${OPEN_SF_ROOT}/.csu_ai_0_runtime_source"
    cleanup_runtime_stage
    trap - EXIT INT TERM
}

if [[ ! -f "${OPEN_SF_ROOT}/pyproject.toml" || ! -f "${OPEN_SF_ROOT}/uv.lock" ]]; then
    install_runtime_source
else
    echo "[CSU-AI-0] Reusing installed CSU-AI-0 runtime source: ${OPEN_SF_ROOT}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${XPOLICYLAB_ROOT}/../.cache/uv}"
mkdir -p "${UV_CACHE_DIR}"

echo "[CSU-AI-0] OPEN_SF_ROOT=${OPEN_SF_ROOT}"
echo "[CSU-AI-0] UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "[CSU-AI-0] UV_BIN=${UV_BIN}"

cd "${OPEN_SF_ROOT}"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" sync --frozen
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" pip install -e .
"${UV_BIN}" pip install -e "${XPOLICYLAB_ROOT}"

chmod +x "${OPEN_SF_ROOT}/.venv/bin/python"* 2>/dev/null || true
if ! "${OPEN_SF_ROOT}/.venv/bin/python" -c "import jax, openpi, XPolicyLab; print('CSU-AI-0 runtime ok')"; then
    echo "[CSU-AI-0][ERROR] runtime import verification failed in ${OPEN_SF_ROOT}/.venv" >&2
    exit 1
fi

echo "[CSU-AI-0] Installation finished."
echo "[CSU-AI-0] Activate: source ${OPEN_SF_ROOT}/.venv/bin/activate"
