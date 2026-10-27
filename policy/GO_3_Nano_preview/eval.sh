#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"
usage() {
    echo 'Usage: eval.sh [--dry-run]'
    echo '   or: eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env> [--dry-run]'
    echo 'With no positional arguments, use configs/runtime.conf.'
}
dry_run=false
args=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=true ;;
        --help|-h) usage; exit 0 ;;
        *) args+=("$argument") ;;
    esac
done
set -- "${args[@]}"
source "$POLICY_DIR/configs/runtime.conf"
if (( $# == 0 )); then
    bench_name="$EVAL_BENCH"; task_name="$EVAL_TASK"; ckpt_name="$CKPT_PATH"
    env_cfg_type="$EVAL_ENV_CFG_TYPE"; action_type="$EVAL_ACTION_TYPE"; seed="$EVAL_SEED"
    policy_gpu_id="$POLICY_GPU_ID"; env_gpu_id="$EVAL_GPU_ID"
    policy_conda_env="$POLICY_ENV"; eval_env_conda_env="$EVAL_ENV"
elif (( $# == 10 )); then
    bench_name=$1; task_name=$2; ckpt_name=$3; env_cfg_type=$4; action_type=$5; seed=$6
    policy_gpu_id=$7; env_gpu_id=$8; policy_conda_env=$9; eval_env_conda_env=${10}
else
    usage >&2; exit 2
fi
for name in bench_name task_name ckpt_name env_cfg_type action_type seed policy_gpu_id env_gpu_id policy_conda_env eval_env_conda_env; do
    [[ -n "${!name}" ]] || { echo "Missing $name: fill configs/runtime.conf or pass all ten positional arguments." >&2; exit 2; }
done
[[ "${bench_name,,}" == robotwin && "$action_type" == joint ]] || { echo 'Supported: RoboTwin / joint' >&2; exit 2; }
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
if "$dry_run"; then
    printf '[SERVER COMMAND] '
    printf '%q ' bash "$SERVER_SCRIPT" "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" "$policy_gpu_id" "$policy_conda_env" AUTO_PORT
    printf '\n[CLIENT COMMAND] '
    printf '%q ' bash "$CLIENT_SCRIPT" "$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" "$env_gpu_id" "$eval_env_conda_env" "$additional_info" AUTO_PORT "$policy_server_ip"
    printf '\n'
    exit 0
fi
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

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
