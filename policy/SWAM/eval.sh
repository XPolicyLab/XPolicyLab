#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?seed}
policy_gpu_id=${7:?policy_gpu_id}
env_gpu_id=${8:?env_gpu_id}
policy_conda_env=${9:?policy_conda_env}
eval_env_conda_env=${10:?eval_env_conda_env}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[MAIN] starting SWAM server; cold start can take 8-10 minutes"
bash "${SERVER_SCRIPT}" "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
  "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
  "${policy_server_port}" "${policy_server_ip}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" \
  "${SERVER_PID}" "SWAM policy server" 1200

bash "${CLIENT_SCRIPT}" "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
  "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
  "${additional_info}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
