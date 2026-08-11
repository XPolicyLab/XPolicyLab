#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
env_gpu_id=${8:?env_gpu_id required}
policy_env=${9:?policy env/python/venv required}
eval_env_conda_env=${10:?eval env required}

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

echo "[MAIN] start G05 policy server on ${policy_server_ip}:${policy_server_port}"
bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
  "${policy_gpu_id}" "${policy_env}" "${policy_server_port}" "${policy_server_ip}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "G05 policy server" 1200

echo "[MAIN] start env client"
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
  "${env_gpu_id}" "${eval_env_conda_env}" "${additional_info}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
