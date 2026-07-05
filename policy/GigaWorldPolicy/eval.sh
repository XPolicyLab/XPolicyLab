#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 11 ]]; then
  echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>" >&2
  exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
env_gpu_id=$9
policy_conda_env=${10}
eval_env_conda_env=${11}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"
policy_name="$(basename "${SCRIPT_DIR}")"

policy_server_port="${POLICY_SERVER_PORT:-$(bash "${UTILS_DIR}/get_free_port.sh")}"
policy_server_host="${POLICY_SERVER_HOST:-${GIGAWORLD_POLICY_SERVER_HOST:-localhost}}"
policy_server_ip="${POLICY_SERVER_IP:-${GIGAWORLD_POLICY_SERVER_IP:-}}"
if [[ -z "${policy_server_ip}" ]]; then
  case "${policy_server_host}" in
    0.0.0.0|::) policy_server_ip="localhost" ;;
    *) policy_server_ip="${policy_server_host}" ;;
  esac
fi

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
ready_timeout="${POLICY_SERVER_READY_TIMEOUT:-${GIGAWORLD_SERVER_READY_TIMEOUT:-360}}"
startup_wait="${GIGAWORLD_SERVER_STARTUP_WAIT:-5}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "[MAIN] kill server ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[MAIN] policy=${policy_name}"
echo "[MAIN] policy_gpu=${policy_gpu_id}, env_gpu=${env_gpu_id}"
echo "[MAIN] start server bind=${policy_server_host}:${policy_server_port}"
echo "[MAIN] client will connect to ${policy_server_ip}:${policy_server_port}"

bash "${SERVER_SCRIPT}" \
  "${bench_name}" \
  "${task_name}" \
  "${ckpt_name}" \
  "${env_cfg_type}" \
  "${expert_data_num}" \
  "${action_type}" \
  "${seed}" \
  "${policy_gpu_id}" \
  "${policy_conda_env}" \
  "${policy_server_port}" \
  "${policy_server_host}" &
SERVER_PID=$!

if [[ -f "${UTILS_DIR}/wait_for_policy_server.sh" ]]; then
  bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" \
    "${policy_server_port}" \
    "${SERVER_PID}" \
    "${policy_name} server" \
    "${ready_timeout}"
else
  sleep "${startup_wait}"
fi

echo "[MAIN] start env client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
  "${bench_name}" \
  "${task_name}" \
  "${ckpt_name}" \
  "${env_cfg_type}" \
  "${action_type}" \
  "${seed}" \
  "${env_gpu_id}" \
  "${eval_env_conda_env}" \
  "${additional_info}" \
  "${policy_server_port}" \
  "${policy_server_ip}"

echo "[MAIN] eval finished"
