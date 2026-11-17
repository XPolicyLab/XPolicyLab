#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}"
python -m pip install "https://github.com/org2AI/worldterminal-sdk/releases/download/v0.1.0/worldterminal_sdk-0.1.0-py3-none-any.whl#sha256=2f08062f2ca6c6ce875d47e632ee8e4a029dbabbe1adf6864b350212b64169f9"
python -c 'from worldterminal_sdk import Client; print("WorldTerminal SDK ready")'
