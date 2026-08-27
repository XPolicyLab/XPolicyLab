#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_host=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
yaml_file="${MEITUAN_ROBOTICS_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"
if [[ "${yaml_file}" != /* ]]; then
    yaml_file="${SCRIPT_DIR}/${yaml_file}"
fi

client_args=("${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" "${policy_server_port}" "${bench_name}" "${task_name}" "${env_cfg_type}" Meituan_Robotics_0 "${additional_info}" "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_host}")
bash "${UTILS_DIR}/setup_env_client.sh" "${client_args[@]}"
