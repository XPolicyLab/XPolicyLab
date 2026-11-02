#!/bin/bash
set -euo pipefail

# Install XPolicyLab (editable) plus the wall-x serving stack.
#
# WALLX_REPO points at a wall-x checkout (default: <bench root>/wall-x inside
# the workspace). The upstream requirements.txt is authoritative for the
# serving dependencies; this script only makes both packages importable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
WALLX_REPO="${WALLX_REPO:-${BENCH_ROOT}/wall-x}"

python -m pip install -e "${XPL_ROOT}"

if [[ -d "${WALLX_REPO}" ]]; then
    echo "[install] installing wall-x from ${WALLX_REPO}"
    # --no-build-isolation: wall-x compiles its CUDA operators against the installed torch
    MAX_JOBS="${MAX_JOBS:-8}" python -m pip install --no-build-isolation -e "${WALLX_REPO}"
else
    echo "[install] WALLX_REPO=${WALLX_REPO} not found; skipping wall-x." >&2
    echo "[install] Clone wall-x (see README.md, Installation) and re-run, or set WALLX_REPO." >&2
fi
