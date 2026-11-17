#!/bin/bash
set -euo pipefail

# XBrain_v1 installs XPolicyLab itself so
# imports such as `XPolicyLab.policy.XBrain_v1.model` and `client_server.ws`
# resolve in the policy environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"

PYTHON_BIN="${XBRAIN_PYTHON:-${PYTHON_BIN:-}}"
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "[XBrain_v1][ERROR] Python interpreter not found. Set XBRAIN_PYTHON=/path/to/python." >&2
    exit 1
fi

echo "[XBrain_v1] python=${PYTHON_BIN}"
if [[ ! -f "${RUNTIME_DIR}/setup.py" ]]; then
    echo "[XBrain_v1][ERROR] bundled inference runtime is missing: ${RUNTIME_DIR}" >&2
    exit 2
fi
"${PYTHON_BIN}" -m pip install -r "${SCRIPT_DIR}/requirements-inference.txt"
"${PYTHON_BIN}" -m pip install -e "${RUNTIME_DIR}"
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"
"${PYTHON_BIN}" - <<'PY'
from xbrain_v1_runtime import runtime_info

print("[XBrain_v1] runtime ready:", runtime_info())
PY
