#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
policy_venv="${1:-$SCRIPT_DIR/.venv}"
policy_python="${EMBODIEDRSI_PYTHON:-python3}"
"$policy_python" -c 'import sys; assert sys.version_info >= (3, 12), "Python >=3.12 required (numpy==2.5.3)"'
"$policy_python" -m venv "$policy_venv"
"$policy_venv/bin/python" -m pip --isolated install --no-cache-dir --index-url https://pypi.org/simple \
    -e "$XPL_ROOT" -r "$SCRIPT_DIR/runtime/requirements.txt"
echo "[INSTALL] Policy environment ready: $policy_venv"
echo "[INSTALL] Real evaluation also requires Docker/CLI image, credentials and the official RoboDojo workspace. See README.md."
