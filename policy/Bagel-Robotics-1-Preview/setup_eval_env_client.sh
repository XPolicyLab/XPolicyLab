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

# The shared debug launcher requires Conda. Support a policy venv directly for
# this adapter's debug checks while keeping the normal simulator path intact.
debug_python=""
if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    debug_python="${eval_env_conda_env}/bin/python"
elif [[ -x "${SCRIPT_DIR}/${eval_env_conda_env}/bin/python" ]]; then
    debug_python="${SCRIPT_DIR}/${eval_env_conda_env}/bin/python"
fi
if [[ "${EVAL_ENV_TYPE:-}" == "debug" && -n "${debug_python}" ]]; then
    debug_episodes="${DEBUG_EVAL_EPISODES:-10}"
    if [[ ! "${debug_episodes}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] DEBUG_EVAL_EPISODES must be a positive integer" >&2
        exit 2
    fi
    export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec env CUDA_VISIBLE_DEVICES="" "${debug_python}" "${UTILS_DIR}/debug_env_client.py" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "${policy_name}" \
        --protocol ws \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_batch false \
        --eval_episode_num "${debug_episodes}"
fi

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
