#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
policy_gpu_id=${7:?}
env_gpu_id=${8:?}
policy_conda_env=${9:?}
eval_env_conda_env=${10:?}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/inference_launch_common.sh"
wsp_select_deploy "${bench_name}"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip=localhost
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[CLEANUP] kill server process group ${SERVER_PID}"
        kill -- "-${SERVER_PID}" 2>/dev/null \
            || kill "${SERVER_PID}" 2>/dev/null \
            || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_command=(bash "${SCRIPT_DIR}/setup_eval_policy_server.sh"
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}"
    "${action_type}" "${seed}" "${policy_gpu_id}"
    "${policy_conda_env}" "${policy_server_port}" "${policy_server_ip}")
client_command=(bash "${SCRIPT_DIR}/setup_eval_env_client.sh"
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}"
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}"
    "${additional_info}" "${policy_server_port}" "${policy_server_ip}")

if [[ "${WSP_LAUNCH_DRY_RUN:-false}" == true ]]; then
    "${server_command[@]}"
    "${client_command[@]}"
    exit 0
fi

setsid "${server_command[@]}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" \
    "${policy_name} server" 1800

"${client_command[@]}"

echo "[MAIN] eval finished"
