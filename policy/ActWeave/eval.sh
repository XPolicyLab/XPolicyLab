#!/bin/bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
policy_gpu_id=${7:?}
env_gpu_id=${8:?}
policy_env=${9:?}
eval_env=${10:?}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xpl_root="$(cd "${script_dir}/../.." && pwd -P)"
server_port=$(bash "${xpl_root}/utils/get_free_port.sh")
server_pid=

cleanup() {
    status=$?
    trap - EXIT
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}"
        if wait "${server_pid}"; then :; fi
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bash "${script_dir}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_env}" "${server_port}" &
server_pid=$!
bash "${xpl_root}/utils/wait_for_policy_server.sh" localhost "${server_port}" "${server_pid}" "ActWeave policy" 1200
bash "${script_dir}/setup_eval_env_client.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env}" \
    "ckpt_name=${ckpt_name},action_type=${action_type}" "${server_port}" localhost
