#!/usr/bin/env bash
# Python packages the bridge-mode policy server needs on top of the Isaac environment: websockets for the server,
# transformers for the monocular depth model (torch comes with Isaac Sim).
# Installed into .cache/policy-deps and put first on PYTHONPATH by policy/RoboBigWater/setup_eval_policy_server.sh.
set -euo pipefail
source "$(dirname "$0")/sim_env.sh"
unset PIP_NO_INDEX  # sim_env.sh keeps the simulator offline; this script needs the index
export PATH=$HOME/.local/bin:$PATH
DEPS=$ROBOSHELL_ROOT/.cache/policy-deps
mkdir -p "$DEPS"
# websockets with its dependencies; the depth-model packages without theirs (torch, numpy, tokenizers, regex, safetensors,
# tqdm, requests and pyyaml come with the Isaac environment and must not be shadowed by other versions)
pipi() { "$ISAAC_PYTHON" -m pip install --target "$DEPS" --upgrade ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"} "$@" 2>&1 | tail -1 \
  || uv pip install --python "$ISAAC_PYTHON" --target "$DEPS" ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"} "$@" 2>&1 | tail -1; }
pipi "websockets>=13"
pipi --no-deps "transformers>=4.45,<5" "huggingface_hub>=0.30,<1"
PYTHONPATH="$DEPS" "$ISAAC_PYTHON" -c "import websockets, websockets.asyncio.server; print('websockets', websockets.__version__)"
PYTHONPATH="$DEPS" "$ISAAC_PYTHON" -c "import transformers; print('transformers', transformers.__version__)"
