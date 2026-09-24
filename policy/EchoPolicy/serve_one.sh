#!/usr/bin/env bash
set -euo pipefail
# Usage: ... <gpu> <public-port> <checkpoint> [internal-port]
GPU=${1:?GPU index required}
PORT=${2:?Public port required}
CHECKPOINT=${3:?Pi05 checkpoint required}
BACKEND_PORT=${4:-$((PORT + 10000))}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XPL_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export ECHOPOLICY_ROOT=${ECHOPOLICY_ROOT:-$SCRIPT_DIR/echopolicy}
export ECHOPOLICY_RUNTIME_ROOT=${ECHOPOLICY_RUNTIME_ROOT:-$SCRIPT_DIR/runtime}
: "${VLM_API_KEY:?Set VLM_API_KEY in the environment}"
PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH="$ECHOPOLICY_ROOT:$(dirname "$XPL_ROOT"):$XPL_ROOT:$XPL_ROOT/policy/Pi_05/openpi/src:$XPL_ROOT/policy/Pi_05/openpi/packages/openpi-client/src:${PYTHONPATH:-}"
export XDG_CACHE_HOME="$ECHOPOLICY_RUNTIME_ROOT/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" OPENPI_DATA_HOME="$XDG_CACHE_HOME/openpi"
export JAX_COMPILATION_CACHE_DIR="$XDG_CACHE_HOME/jax" TMPDIR="$ECHOPOLICY_RUNTIME_ROOT/tmp"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_HOME="$XDG_CACHE_HOME/torch"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$ECHOPOLICY_RUNTIME_ROOT/logs/policy-$PORT"
"$PYTHON_BIN" "$XPL_ROOT/setup_policy_server.py" --config_path "$SCRIPT_DIR/deploy.yml" \
  --overrides host=127.0.0.1 port="$BACKEND_PORT" ckpt_name="$CHECKPOINT" \
    bench_name="${BENCH_NAME:-RoboDojo}" task_name="${TASK_NAME:-cover_blocks}" \
    env_cfg_type="${ENV_CFG_TYPE:-arx_x5}" action_type="${ACTION_TYPE:-joint}" seed="${POLICY_SEED:-0}" \
  > "$ECHOPOLICY_RUNTIME_ROOT/logs/pi05-$PORT.log" 2>&1 &
BACKEND_PID=$!
PROXY_PID=""
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$PROXY_PID" ]] || kill "$PROXY_PID" 2>/dev/null || true
  kill "$BACKEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  [[ -z "$PROXY_PID" ]] || wait "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON_BIN" - "$BACKEND_PORT" "$BACKEND_PID" <<'PY'
import os, sys, time
from pathlib import Path
from websockets.sync.client import connect
port, pid = map(int, sys.argv[1:])
for _ in range(1800):
    os.kill(pid, 0)
    if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
        raise RuntimeError("Pi05 exited during startup; inspect backend log")
    try:
        with connect(f'ws://127.0.0.1:{port}', open_timeout=1): break
    except OSError: time.sleep(1)
else: raise TimeoutError('Pi05 startup exceeded 30 minutes')
PY
"$PYTHON_BIN" -m vlm_orchestrator.cli \
  --host "${ORCHESTRATOR_HOST:-0.0.0.0}" --port "$PORT" --vla-host 127.0.0.1 --vla-port "$BACKEND_PORT" \
  --mode subgoal --vla-candidates 16 --vlm-model "${VLM_MODEL:-gemini-3.8-flash}" \
  --vlm-thinking-level "${VLM_THINKING_LEVEL:-medium}" \
  --vlm-base-url "${VLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}" \
  --log-dir "$ECHOPOLICY_RUNTIME_ROOT/logs/policy-$PORT" "${@:5}" &
PROXY_PID=$!
# Exit and clean up both processes if either one exits.
wait -n "$BACKEND_PID" "$PROXY_PID"
