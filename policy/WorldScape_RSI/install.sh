#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${SCRIPT_DIR}/worldscape-policy[server]" -e "${XPL_ROOT}"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt" -r "${SCRIPT_DIR}/requirements-runtime.txt"
python "${SCRIPT_DIR}/preflight.py" --check-runtime
echo "Bind public WAM checkpoints and ROBODOJO_ROOT; see INSTALLATION.md."
