#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_conda_env=${9}
eval_env_conda_env=${10}

MACH_EMBODIED_DEX_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# XPolicyLab reads deploy.yml before activating the client environment. Use the
# policy environment installed by install.sh for that launcher step; the
# official helper activates eval_env_conda_env before starting the real client.
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    export PATH="${policy_conda_env}/bin:${PATH}"
elif [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    export PATH="${eval_env_conda_env}/bin:${PATH}"
fi
XPL_ROOT="$(cd "${MACH_EMBODIED_DEX_EVAL_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

bash "${MACH_EMBODIED_DEX_EVAL_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
    "${policy_server_port}" &
SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" localhost "${policy_server_port}" \
    "${SERVER_PID}" "Policy server" 1200
bash "${MACH_EMBODIED_DEX_EVAL_DIR}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "ckpt_name=${ckpt_name},action_type=${action_type},task_config=${ROBOTWIN_TASK_CONFIG:-demo_clean},test_num=${ROBOTWIN_TEST_NUM:-100},expert_check=${ROBOTWIN_EXPERT_CHECK:-true}" \
    "${policy_server_port}" localhost
echo "[MAIN] eval finished"
