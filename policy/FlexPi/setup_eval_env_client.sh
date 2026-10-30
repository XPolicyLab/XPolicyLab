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
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo -e "\033[34m[CLIENT] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[34m[CLIENT] server=${policy_server_ip}:${policy_server_port}\033[0m"

# Scope CUDA to this client process; setup_env_client.sh dispatches debug / sim
# based on EVAL_ENV_TYPE (default: sim).
exec env \
    CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    bash "${UTILS_DIR}/setup_env_client.sh" \
        "${UTILS_DIR}" \
        "${yaml_file}" \
        "${eval_env_conda_env}" \
        "${policy_server_port}" \
        "${bench_name}" \
        "${task_name}" \
        "${env_cfg_type}" \
        "${policy_name}" \
        "${additional_info}" \
        "${BENCH_ROOT}" \
        "${seed}" \
        "${env_gpu_id}" \
        "${policy_server_ip}"
