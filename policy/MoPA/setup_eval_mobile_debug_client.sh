#!/bin/bash
set -euo pipefail

env_gpu_id=${1:?env gpu id required}
eval_env_conda_env=${2:?evaluation conda env required}
policy_server_port=${3:?policy server port required}
policy_server_ip=${4:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/resolve_python.sh"
PY="$(resolve_python "${eval_env_conda_env}")"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
export PYTHONPATH="${WORKSPACE_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"

args=(
    --host "${policy_server_ip}"
    --port "${policy_server_port}"
    --episodes "${MOPA_DEBUG_EPISODES:-1}"
    --episode-step-limit "${MOPA_DEBUG_STEPS:-32}"
    --batch-size "${MOPA_DEBUG_BATCH_SIZE:-3}"
    --action-key-style "${MOPA_ACTION_KEY_STYLE:-singular}"
)
if [[ "${MOPA_DEBUG_BATCH:-false}" == "true" ]]; then
    args+=(--eval-batch)
fi
if [[ "${MOPA_DEBUG_ENCODED:-false}" == "true" ]]; then
    args+=(--encoded)
fi

exec env CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    "${PY}" "${SCRIPT_DIR}/mobile_debug_client.py" "${args[@]}"
