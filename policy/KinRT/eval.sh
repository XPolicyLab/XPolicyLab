#!/usr/bin/env bash
set -euo pipefail
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"

if ! command -v conda >/dev/null 2>&1; then
  for conda_candidate in \
    "${CONDA_EXE:-}" \
    /opt/conda/bin/conda \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda"; do
    if [[ -n "${conda_candidate}" && -x "${conda_candidate}" ]]; then
      export PATH="$(dirname "${conda_candidate}"):${PATH}"
      break
    fi
  done
fi

if [[ $# -lt 10 ]]; then
  echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>" >&2
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
policy_uv_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "[MAIN] kill server ${SERVER_PID}"
    kill -TERM -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[MAIN] start KinRT server, port=${policy_server_port}"
setsid bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
  "${seed}" "${policy_gpu_id}" "${policy_uv_env}" "${policy_server_port}" "${policy_server_ip}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
  "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
  "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
  "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" "${additional_info}" \
  "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
