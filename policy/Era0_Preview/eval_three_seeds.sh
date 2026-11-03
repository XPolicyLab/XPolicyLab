#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
resolve_python "${POLICY_ENV:-${POLICY_DIR}/.venv}"
exec "${PYTHON_BIN}" "${POLICY_DIR}/evaluate.py" "$@"
