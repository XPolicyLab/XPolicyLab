#!/bin/bash
set -euo pipefail

# Check transport and action shapes with synthetic mobile observations.
bench_name=${1:-mobile}
task_name=${2:-mobile_debug}
ckpt_name=${3:-mopa}
env_cfg_type=${4:-m92uw}
action_type=${5:-joint}
seed=${6:-0}
policy_gpu_id=${7:-0}
env_gpu_id=${8:-0}
policy_conda_env=${9:-mopa}
eval_env_conda_env=${10:-mopa}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[MAIN] start MoPA extended mobile server on ${policy_server_ip}:${policy_server_port}"
MOPA_ACTION_KEY_STYLE="${MOPA_ACTION_KEY_STYLE:-singular}" MOBILE_ACTION_CONTRACT=extended90 \
    bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
    "${policy_server_port}" "${policy_server_ip}" &
server_pid=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" \
    "${policy_server_port}" "${server_pid}" "MoPA extended mobile server" 1800

echo "[MAIN] run extended mobile protocol client"
bash "${SCRIPT_DIR}/setup_eval_mobile_debug_client.sh" \
    "${env_gpu_id}" "${eval_env_conda_env}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] extended mobile protocol evaluation finished"
