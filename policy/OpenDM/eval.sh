#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
    echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> [policy_conda_env] [eval_env_conda_env]" >&2
    exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
default_conda_env="${CONDA_DEFAULT_ENV:-}"
policy_conda_env=${9:-${default_conda_env}}
eval_env_conda_env=${10:-${policy_conda_env}}

if [[ -z "${policy_conda_env}" || -z "${eval_env_conda_env}" ]]; then
    echo "Policy and evaluation conda environments must be provided." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="${POLICY_SERVER_HOST:-localhost}"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start OpenDM policy server on ${policy_server_ip}:${policy_server_port}"
bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    "${policy_gpu_id}" "${policy_conda_env}" "${policy_server_port}" "${policy_server_ip}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "OpenDM policy server" 1200

echo "[MAIN] start environment client"
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
    "${env_gpu_id}" "${eval_env_conda_env}" "${additional_info}" \
    "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
