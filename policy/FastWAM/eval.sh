#!/bin/bash
set -euo pipefail

# Contract: 11 positional args. `task_name` is the simulator task; `ckpt_name`
# resolves the checkpoint directory and may differ (e.g. `cotrain`).
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
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] kill server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo -e "\033[32m[MAIN] start server, policy_server_port=${policy_server_port}\033[0m"
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
    "${policy_server_ip}" &
SERVER_PID=$!
echo -e "\033[32m[MAIN] server PID=${SERVER_PID}\033[0m"

# Wait for the server to either open the port or exit early.
for _ in $(seq 1 180); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo -e "\033[31m[ERROR] Policy server exited before opening port ${policy_server_port}.\033[0m" >&2
        exit 1
    fi
    if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${policy_server_ip}', int('${policy_server_port}'))); s.close()" >/dev/null 2>&1; then
        echo -e "\033[32m[MAIN] server is ready on port ${policy_server_port}\033[0m"
        break
    fi
    sleep 2
done

echo -e "\033[32m[MAIN] start client, server=${policy_server_ip}:${policy_server_port}\033[0m"
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

echo -e "\033[33m[MAIN] eval finished\033[0m"
