#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == Linux ]] || { echo '[error] Linux is required' >&2; exit 1; }
MODE="${1:-install}"
[[ "$MODE" == install || "$MODE" == --check ]] || { echo 'usage: bash install.sh [--check]' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"

# Same interpreter as install.sh: PYTHON_BIN wins when the outer installer (or the
# caller) selected one, so the import check below runs in the environment that
# actually holds the packages.
# shellcheck source=../select_python.sh
source "$(cd "$SCRIPT_DIR/.." && pwd)/select_python.sh"
_xpl_pick_python "$XPL_ROOT" || exit 1
PYTHON="$XPL_PYTHON"
ENV_DIR="$XPL_ENV_DIR"

[[ -f "$XPL_ROOT/pyproject.toml" && -f "$SCRIPT_DIR/runtime/serve_geact.py" ]] || {
  echo '[error] XPolicyLab or bundled GEAct runtime is missing' >&2; exit 1;
}

if [[ "$MODE" == install ]]; then
  _xpl_check_python_version || exit 1
  "$PYTHON" -m ensurepip --upgrade >/dev/null
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-infer.txt"
  "$PYTHON" -m pip install -e "$XPL_ROOT"
fi

[[ -x "$PYTHON" ]] || { echo "[error] environment missing: $PYTHON" >&2; exit 1; }
PYTHONPATH="$BENCH_ROOT:$SCRIPT_DIR/runtime" "$PYTHON" -c \
  'import torch, cv2, msgpack, yaml, websockets; from XPolicyLab.policy.go3_agent_preview.wam.model import Model; from robodojo_geact.model.geact.build_models import build_geact_task_flow; print("GEAct imports OK; CUDA:", torch.cuda.is_available())'
echo "[ok] environment=$PYTHON"
echo '[note] Checkpoint, statistics, VAE and Qwen assets are read from their configured paths.'
