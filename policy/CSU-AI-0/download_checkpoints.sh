#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CSU_AI_0_DOWNLOAD_PYTHON:-$(command -v python3 || command -v python)}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "[CSU-AI-0][ERROR] Python 3 is required; set CSU_AI_0_DOWNLOAD_PYTHON" >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/download_checkpoints.py" "$@"
