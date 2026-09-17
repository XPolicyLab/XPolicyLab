#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" && ( -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == base ) ]]; then
    echo "Activate a dedicated conda/venv environment before running install.sh." >&2
    exit 1
fi
python -m pip install -e "${SCRIPT_DIR}/mopa"
python -m pip install -e "${XPL_ROOT}"
