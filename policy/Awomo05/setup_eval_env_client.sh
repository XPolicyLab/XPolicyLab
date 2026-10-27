#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

if [[ "${EVAL_ENV_TYPE:-sim}" == "debug" && "${eval_env_conda_env}" == "system" ]]; then
    export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${AWOMO05_DEBUG_PYTHON:-python}" "${UTILS_DIR}/debug_env_client.py" \
        --bench_name "${bench_name}" --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" --policy_name "${policy_name}" \
        --protocol ws --host "${policy_server_ip}" --port "${policy_server_port}" \
        --eval_batch true --eval_episode_num "${AWOMO05_DEBUG_EPISODES:-1}"
fi

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
