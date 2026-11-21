#!/usr/bin/env bash
# Same-machine evaluation, aligned with policy/demo_policy/eval.sh.
#
# Accepts the standard ten arguments:
#   bench task ckpt env_cfg action seed policy_gpu env_gpu policy_env eval_env
# and the eleven-argument form that inserts expert_num after env_cfg (the older
# shared entry-point convention). It starts the policy server, waits for it,
# runs the simulator client, and stops the server on exit.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
         <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
       bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <expert_num> \
         <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> \
         <eval_env_conda_env>
USAGE
}

case $# in
  10) bench_name=$1 task_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6 \
        policy_gpu_id=$7 env_gpu_id=$8 policy_conda_env=$9 eval_env_conda_env=${10} ;;
  11) bench_name=$1 task_name=$2 ckpt_name=$3 env_cfg_type=$4 expert_num=$5 action_type=$6 \
        seed=$7 policy_gpu_id=$8 env_gpu_id=$9 policy_conda_env=${10} eval_env_conda_env=${11} ;;
  *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"
for script in "$SERVER_SCRIPT" "$CLIENT_SCRIPT"; do
  [[ -f "$script" ]] || { echo "[error] missing $script" >&2; exit 1; }
done

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    echo "[MAIN] kill server ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
    "${seed}" "${policy_gpu_id}" "${policy_conda_env}" "${policy_server_port}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
    "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" "${additional_info}" \
    "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
