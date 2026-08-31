#!/bin/bash
set -euo pipefail

# Cluster login environments commonly export HTTP(S)/ALL_PROXY.  websockets
# otherwise routes loopback traffic through that proxy (for example port
# 9977), even though the policy server and simulator client share a node.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"

bench_name=${1:?Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>}
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
port=$(bash "${XPL_ROOT}/utils/get_free_port.sh")
server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
    "${port}" 127.0.0.1 &
server_pid=$!

bash "${XPL_ROOT}/utils/wait_for_policy_server.sh" 127.0.0.1 "${port}" "${server_pid}" "Policy server" 1200

bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "ckpt_name=${ckpt_name},action_type=${action_type}" "${port}" 127.0.0.1

echo "[MAIN] eval finished"
