#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${CSU_CHECKPOINT_ROOT:-${SCRIPT_DIR}/checkpoints/CSU-AI-0-v1}"
PYTHON_BIN="${CSU_DOWNLOAD_PYTHON:-$(command -v python3 || command -v python)}"

[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
    echo "[CSU-AI-0][ERROR] Python was not found; set CSU_DOWNLOAD_PYTHON" >&2
    exit 1
}

args=(--root "${BUNDLE_DIR}")
if [[ "${CSU_DOWNLOAD_DRY_RUN:-0}" == "1" ]]; then
    args+=(--dry-run)
fi
if [[ "${CSU_REPLACE_INVALID_CHECKPOINTS:-0}" == "1" ]]; then
    args+=(--replace-invalid)
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/download_checkpoints.py" "${args[@]}"
