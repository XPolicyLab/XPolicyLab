#!/usr/bin/env bash
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

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export NO_PROXY="${NO_PROXY},${policy_server_ip}"
export no_proxy="${no_proxy},${policy_server_ip}"

# shellcheck source=/dev/null
source "${HWGOAL_XPL_ROOT}/utils/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)"

echo "[CLIENT] policy=HumanWAM-Goal, task=${task_name}, mode=${eval_env_mode}, server=${policy_server_ip}:${policy_server_port}"

goalwam_select_python "${eval_env_conda_env}"

if [[ "${eval_env_mode}" == "debug" ]]; then
    exec env \
        CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
        "${HWGOAL_EXEC_PYTHON}" "${HWGOAL_XPL_ROOT}/utils/debug_env_client.py" \
            --bench_name "${bench_name}" \
            --task_name "${task_name}" \
            --env_cfg_type "${env_cfg_type}" \
            --policy_name HumanWAM_Goal \
            --host "${policy_server_ip}" \
            --port "${policy_server_port}" \
            --protocol ws \
            --eval_batch false
elif [[ "${eval_env_mode}" != "sim" ]]; then
    echo "[CLIENT][ERROR] HumanWAM-Goal supports EVAL_ENV_TYPE=sim or debug" >&2
    exit 2
fi

exec env \
    CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    "${HWGOAL_EXEC_PYTHON}" "${HWGOAL_ADAPTER_DIR}/launch.py" client \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
        --port "${policy_server_port}" --host "${policy_server_ip}"
