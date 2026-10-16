#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
    echo "usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>" >&2
    exit 2
fi
: "${CODEX_API_KEY:?Set CODEX_API_KEY before running SenseNova-RoboRSI evaluation}"
export CODEX_API_KEY

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

if ! command -v setsid >/dev/null 2>&1; then
    echo "[MAIN][ERROR] setsid is required to clean up the policy and Codex processes" >&2
    exit 1
fi

cleanup() {
    local exit_status=$?
    trap - EXIT INT TERM
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] stop policy process group ${SERVER_PID}"
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in {1..200}; do
            if ! kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 0.2
        done
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    return "${exit_status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

policy_server_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
policy_server_ip=localhost
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

echo "[MAIN] start server, policy_server_port=${policy_server_port}"
setsid bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_env}" \
    "${policy_server_port}" "${policy_server_ip}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "${additional_info}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
