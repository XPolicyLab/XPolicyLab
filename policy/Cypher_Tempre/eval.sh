#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_uv_env=${9:-uv}
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

export XPL_POLICY_NAME=Cypher_Tempre
export XPL_DEPLOY_YML="${SCRIPT_DIR}/deploy.yml"
export XPL_CONDITION=Cypher-Tempre-continual
export L4_INSPECT_TRACE_NAMESPACE=xpolicylab-cypher-tempre
export L4_INSPECT_MODEL="${CYPHER_TEMPRE_MODEL:-gpt-6.1-sol}"

# RoboDojo invokes this script from the adapter directory. Leave it before
# any helper that may start Python.
cd "${XPL_ROOT}"

if [[ -n "${L4_INSPECT_DEPTH:-}" && "${L4_INSPECT_DEPTH}" != "off" ]]; then
    echo "[L4][ERROR] RoboDojo_Agent_L4_Inspect is RGB-only; set L4_INSPECT_DEPTH=off or unset it." >&2
    exit 1
fi
unset ROBODOJO_ENABLE_METRIC_DEPTH

BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
export ROBODOJO_ROOT="${ROBODOJO_ROOT:-${BENCH_ROOT}/RoboDojo-eval}"
export L4_INSPECT_REASONING_EFFORT="${CYPHER_TEMPRE_REASONING_EFFORT:-xhigh}"
export L4_INSPECT_KEEP_ALL_IMAGES="${L4_INSPECT_KEEP_ALL_IMAGES:-0}"
export L4_INSPECT_IMAGE_HORIZON="${L4_INSPECT_IMAGE_HORIZON:-2}"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export L4_INSPECT_ACTION_TYPE="${action_type}"
export ROBODOJO_ACTION_TYPE="${action_type}"
if [[ -n "${L4_INSPECT_TRACE_NAMESPACE:-}" ]]; then
    _trace_root="${TMPDIR:-/tmp}/${L4_INSPECT_TRACE_NAMESPACE}-${USER:-$(id -un)}"
else
    _trace_root="${TMPDIR:-/tmp}/xpolicylab-l4-inspect-${USER:-$(id -un)}"
fi
export L4_INSPECT_TRACE_DIR="${L4_INSPECT_TRACE_DIR:-${_trace_root}/${task_name}/seed-${seed}}"
mkdir -p "${L4_INSPECT_TRACE_DIR}"
condition="${XPL_CONDITION:-L4-inspect-local}"
additional_info="condition=${condition},planner=${L4_INSPECT_PLANNER:-astra},model=${L4_INSPECT_MODEL:-},depth=off,keep_all_images=${L4_INSPECT_KEEP_ALL_IMAGES:-1},image_horizon=${L4_INSPECT_IMAGE_HORIZON:-2},max_llm_calls=${L4_INSPECT_MAX_LLM_CALLS:-100},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM -- -"${SERVER_PID}" 2>/dev/null \
            || kill "${SERVER_PID}" 2>/dev/null \
            || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start L4 inspect policy server, port=${policy_server_port}"
if command -v setsid >/dev/null 2>&1; then
    setsid bash "${SERVER_SCRIPT}" \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_uv_env}" \
        "${policy_server_port}" "${policy_server_ip}" &
else
    bash "${SERVER_SCRIPT}" \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_uv_env}" \
        "${policy_server_port}" "${policy_server_ip}" &
fi
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" \
    "Policy server" 1200

echo "[MAIN] start L4 inspect client, server=${policy_server_ip}:${policy_server_port}"
bash "${CLIENT_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "${additional_info}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] L4 inspect eval finished"
