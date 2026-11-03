#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  python_bin="${SCRIPT_DIR}/.venv/bin/python"
else
  python_bin=python3
fi
exec "${python_bin}" "${SCRIPT_DIR}/download_checkpoint.py" "$@"
