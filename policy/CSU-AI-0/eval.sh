#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
_submission_ckpt=$3
env_cfg_type=$4
_submission_action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
_submission_policy_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
ROUTER_PYTHON="${CSU_ROUTER_PYTHON:-}"
if [[ -z "${ROUTER_PYTHON}" ]]; then
    ROUTER_PYTHON="$(command -v python3 || command -v python)"
fi
ROUTE_AUDIT="${CSU_ROUTE_AUDIT:-${SCRIPT_DIR}/results/routing_audit.jsonl}"

eval "$("${ROUTER_PYTHON}" "${SCRIPT_DIR}/route.py" \
    --task-name "${task_name}" --format shell --audit-jsonl "${ROUTE_AUDIT}")"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
additional_info="ckpt_name=QRouter,action_type=${CSU_TARGET_ACTION_TYPE},submission_policy=CSU-AI-0,selected_expert=${CSU_SELECTED_EXPERT},router_sha256=${CSU_ROUTER_SHA256}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[CSU-AI-0] task=${task_name} expert=${CSU_SELECTED_EXPERT} confidence=${CSU_ROUTE_CONFIDENCE} action_type=${CSU_TARGET_ACTION_TYPE}"
echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" QRouter "${env_cfg_type}" auto \
    "${seed}" "${policy_gpu_id}" auto "${policy_server_port}" \
    "${policy_server_ip}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" \
    "CSU-AI-0 policy server" "${CSU_SERVER_READY_TIMEOUT_SECONDS:-1200}"

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" QRouter "${env_cfg_type}" \
    "${CSU_TARGET_ACTION_TYPE}" "${seed}" "${env_gpu_id}" \
    "${eval_env_conda_env}" "${additional_info}" \
    "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
