#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XPL_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
bash "$XPL_ROOT/policy/Pi_05/install.sh"
uv pip install --python "$XPL_ROOT/policy/Pi_05/openpi/.venv/bin/python" -e "$SCRIPT_DIR/echopolicy"
