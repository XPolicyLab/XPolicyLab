#!/usr/bin/env bash
# Python packages the bridge-mode policy server needs on top of the Isaac environment.
# Installed into .cache/policy-deps and put first on PYTHONPATH by policy/RoboBigWater/setup_eval_policy_server.sh.
set -euo pipefail
source "$(dirname "$0")/sim_env.sh"
unset PIP_NO_INDEX  # sim_env.sh keeps the simulator offline; this script needs the index
export PATH=$HOME/.local/bin:$PATH
DEPS=$ROBOSHELL_ROOT/.cache/policy-deps
mkdir -p "$DEPS"
"$ISAAC_PYTHON" -m pip install --target "$DEPS" --upgrade "websockets>=13" ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"} 2>&1 | tail -2 \
  || uv pip install --python "$ISAAC_PYTHON" --target "$DEPS" "websockets>=13" ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"} 2>&1 | tail -2
PYTHONPATH="$DEPS" "$ISAAC_PYTHON" -c "import websockets, websockets.asyncio.server; print('websockets', websockets.__version__)"
