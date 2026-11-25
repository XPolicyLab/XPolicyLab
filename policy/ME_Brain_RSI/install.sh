#!/usr/bin/env bash
set -euo pipefail

mode=${1:-inference}
if [[ $# -gt 1 ]]; then
    echo "[INSTALL][ERROR] Supply at most one installation mode." >&2
    exit 1
fi
if [[ ${mode} == "--help" ]]; then
    echo "Usage: $0 [inference|code|curobo|train|train-v2]"
    exit 0
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
if [[ ${mode} == "code" ]]; then
    uv venv "${POLICY_DIR}/strategy_env/.venv"
    uv pip install --python "${POLICY_DIR}/strategy_env/.venv/bin/python" \
        -e "${XPL_ROOT}" scipy pillow
    echo "CPU strategy environment: ${POLICY_DIR}/strategy_env"
    echo "Native CUDA policies and DINO/SAM3 need their documented extra dependencies/assets."
    exit 0
fi
if [[ ${mode} == "curobo" ]]; then
    # RoboDojo cb2a37d5 pins this CuRobo v2 revision as a git submodule.
    CUROBO_COMMIT=d17b54ce32cba095c0b000c4c58777075d11de0e
    CUROBO_URL=https://github.com/yuechen0614/curobo.git
    CUROBO_ENV="${ME_BRAIN_CUROBO_ENV:-${POLICY_DIR}/strategy_env/curobo/.venv}"
    if [[ ! -x "${CUROBO_ENV}/bin/python" ]]; then
        uv venv --python 3.11 "${CUROBO_ENV}"
    fi
    CUROBO_PYTHON="${CUROBO_ENV}/bin/python"
    "${CUROBO_PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else "CuRobo requires Python 3.11 in this environment")'
    if [[ -n "${ME_BRAIN_CUROBO_SOURCE:-}" ]]; then
        CUROBO_URL="$("${CUROBO_PYTHON}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True).as_uri())' "${ME_BRAIN_CUROBO_SOURCE}")"
    fi
    uv pip install --python "${CUROBO_PYTHON}" \
        --index-url "${ME_BRAIN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
        'torch==2.7.0+cu128'
    uv pip install --python "${CUROBO_PYTHON}" -e "${XPL_ROOT}" \
        -r "${POLICY_DIR}/tasks/curobo-requirements.txt" 'torch==2.7.0+cu128' \
        "nvidia-curobo[cu12] @ git+${CUROBO_URL}@${CUROBO_COMMIT}"
    uv pip check --python "${CUROBO_PYTHON}"
    "${CUROBO_PYTHON}" "${POLICY_DIR}/core/curobo_runtime.py"
    echo "[INSTALL] Set ME_BRAIN_POLICY_PYTHON=${CUROBO_PYTHON} for CuRobo tasks."
    exit 0
fi
if [[ ${mode} != "inference" && ${mode} != "train" && ${mode} != "train-v2" ]]; then
    echo "[INSTALL][ERROR] Choose inference, code, curobo, train, or train-v2." >&2
    exit 1
fi
cd "${POLICY_DIR}/source"
args=(sync)
if [[ ${mode} != "inference" ]]; then
    args+=(--extra "${mode}")
fi
uv "${args[@]}"
uv pip install --python .venv/bin/python -e "${XPL_ROOT}"
uv run --no-sync focus-vlwa install-transformers-patch
echo "[INSTALL] Strategy environment: ${POLICY_DIR}/source/.venv"
echo "[INSTALL] The environment client imports the bundled source automatically."
