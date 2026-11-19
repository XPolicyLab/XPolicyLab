#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
cd "${SLA_DIR}/openpi"
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --locked --python 3.11
UV_LINK_MODE=copy uv pip install --python .venv/bin/python -e . -e "${XPL_ROOT}"
sla_python uv
"${PYTHON_BIN}" -c 'import XPolicyLab, openpi, pinocchio; print("[SLA] Imports OK")'
