#!/bin/bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
    echo "Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_host=localhost

cleanup() {
    [[ -n "${SERVER_PID:-}" ]] && kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

server_args=("$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "${policy_server_port}" "${policy_server_host}")
bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" "${server_args[@]}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_host}" "${policy_server_port}" "${SERVER_PID}" "Meituan_Robotics_0 policy server" 1200

client_args=("$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" "ckpt_name=$3,action_type=$5" "${policy_server_port}" "${policy_server_host}")
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" "${client_args[@]}"

echo "[MAIN] eval finished"
