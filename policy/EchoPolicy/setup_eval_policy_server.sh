#!/usr/bin/env bash
set -euo pipefail
# Standard XPolicyLab server arguments.
export BENCH_NAME=${1:?} TASK_NAME=${2:?}
checkpoint=${3:?}
export ENV_CFG_TYPE=${4:?} ACTION_TYPE=${5:?} POLICY_SEED=${6:?}
gpu=${7:?}
policy_env=${8:-uv}
port=${9:-17001}
export ORCHESTRATOR_HOST=${10:-0.0.0.0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$policy_env" == uv ]]; then
    export PYTHON_BIN="$SCRIPT_DIR/../Pi_05/openpi/.venv/bin/python"
elif [[ -x "$policy_env/bin/python" ]]; then
    export PYTHON_BIN="$policy_env/bin/python"
elif [[ -x "$policy_env/.venv/bin/python" ]]; then
    export PYTHON_BIN="$policy_env/.venv/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$policy_env"
    export PYTHON_BIN=$(command -v python)
fi
exec bash "$SCRIPT_DIR/serve_one.sh" "$gpu" "$port" "$checkpoint" "${BACKEND_PORT:-$((port + 10000))}"
