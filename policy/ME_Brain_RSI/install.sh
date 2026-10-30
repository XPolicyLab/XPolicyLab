#!/usr/bin/env bash
set -euo pipefail

mode=${1:-inference}
if [[ ${mode} == "code" ]]; then
    POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
    uv venv "${POLICY_DIR}/strategy_env/.venv"
    uv pip install --python "${POLICY_DIR}/strategy_env/.venv/bin/python" \
        -e "${XPL_ROOT}" scipy pillow
    echo "CPU strategy environment: ${POLICY_DIR}/strategy_env"
    echo "Native CUDA policies and DINO/SAM3 need their documented extra dependencies/assets."
    exit 0
fi
if [[ ${mode} == "--help" ]]; then
    echo "Usage: $0 [inference|code|train|train-v2]"
    exit 0
fi
if [[ $# -gt 1 || ${mode} != "inference" && ${mode} != "train" && ${mode} != "train-v2" ]]; then
    echo "[INSTALL][ERROR] Choose inference, train, or train-v2; do not combine LeRobot readers." >&2
    exit 1
fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
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
