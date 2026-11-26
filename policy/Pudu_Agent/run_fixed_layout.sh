#!/usr/bin/env bash
# Run L4 inspect on one layout range. No policy GPU: the adapter serves no VLA, so
# the only inference is the API call and the simulator gets the machine to itself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export XPL_POLICY_NAME=Pudu_Agent
export XPL_DEPLOY_YML="${SCRIPT_DIR}/deploy.yml"
export XPL_CONDITION=L4-inspect-eef
export L4_INSPECT_TRACE_NAMESPACE=xpolicylab-l4-inspect-eef

layout=${1:-0}
env_gpu=${2:-0}
task_name=${3:-arrange_largest_number}
eval_env=${4:-uv}
run_id=${ROBODOJO_RUN_ID:-l4-inspect-${task_name}-layout${layout//,/_}}
policy_name="${XPL_POLICY_NAME:-Pudu_Agent}"

# The model is not defaulted here. It comes from L4_INSPECT_PLANNER, which
# supplies it together with the API surface it is served on; pinning a model
# here would silently win over that and run the default one under the other
# planner's name -- an episode that looks like the condition it is not.
export L4_INSPECT_PLANNER="${L4_INSPECT_PLANNER:-astra}"
export L4_INSPECT_REASONING_EFFORT="${L4_INSPECT_REASONING_EFFORT:-medium}"
export L4_INSPECT_KEEP_ALL_IMAGES="${L4_INSPECT_KEEP_ALL_IMAGES:-0}"
export L4_INSPECT_IMAGE_HORIZON="${L4_INSPECT_IMAGE_HORIZON:-2}"
# Keep a malformed provider response out of Isaac's process. The worker is
# cheap to replace; rebuilding the simulator is not.
export L4_INSPECT_HARD_TIMEOUT_S="${L4_INSPECT_HARD_TIMEOUT_S:-90}"

if [[ -n "${L4_INSPECT_DEPTH:-}" && "${L4_INSPECT_DEPTH}" != "off" ]]; then
    echo "[L4][ERROR] RoboDojo_Agent_L4_Inspect is RGB-only; set L4_INSPECT_DEPTH=off or unset it." >&2
    exit 1
fi
unset ROBODOJO_ENABLE_METRIC_DEPTH

l4_action_type="${L4_INSPECT_ACTION_TYPE:-joint}"
export ROBODOJO_RUN_ID="${run_id}"
export L4_INSPECT_ACTION_TYPE="${l4_action_type}"
export ROBODOJO_ACTION_TYPE="${l4_action_type}"
export ROBODOJO_UNTILED_CAMERAS="${ROBODOJO_UNTILED_CAMERAS:-0}"
export ROBODOJO_PATH_TRACING="${ROBODOJO_PATH_TRACING:-1}"
export ROBODOJO_NUM_ENVS=1
export ROBODOJO_SIM_ENV="${ROBODOJO_ROOT:-${XPL_ROOT}/../RoboDojo-eval}/.venv"
if [[ -n "${L4_INSPECT_TRACE_NAMESPACE:-}" ]]; then
    _trace_root="${TMPDIR:-/tmp}/${L4_INSPECT_TRACE_NAMESPACE}-${USER:-$(id -un)}"
else
    _trace_root="${TMPDIR:-/tmp}/xpolicylab-l4-inspect-${USER:-$(id -un)}"
fi
export L4_INSPECT_TRACE_DIR="${L4_INSPECT_TRACE_DIR:-${_trace_root}/${task_name}/layout-${layout}}"
mkdir -p "${L4_INSPECT_TRACE_DIR}"

echo "[L4] planner=${L4_INSPECT_PLANNER}${L4_INSPECT_MODEL:+ model=${L4_INSPECT_MODEL}} max_llm_calls=${L4_INSPECT_MAX_LLM_CALLS:-100} trace=${L4_INSPECT_TRACE_DIR}"

bash "${XPL_ROOT}/scripts/run_robodojo_layout_range.sh" \
    "${policy_name}" "${task_name}" "${layout}" \
    --policy-gpu "${env_gpu}" --env-gpu "${env_gpu}" \
    --eval-env "${eval_env}"
