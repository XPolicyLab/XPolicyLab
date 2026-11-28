#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
# Install adapter dependencies into an already configured SWAM or RoboTwin environment.
"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}" "$PYTHON_BIN" -c 'from client_server.ws.model_server import PolicyServer; print("XPolicyLab websocket imports passed")'
