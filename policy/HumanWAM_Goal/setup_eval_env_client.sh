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
# shellcheck source=/dev/null
source "${HWGOAL_XPL_ROOT}/utils/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)"

echo "[CLIENT] policy=HumanWAM-Goal, task=${task_name}, mode=${eval_env_mode}, server=${policy_server_ip}:${policy_server_port}"

if [[ "${eval_env_mode}" == "debug" ]]; then
    # XPolicyLab's standard debug client and deploy.py loop.
    exec bash "${HWGOAL_XPL_ROOT}/utils/setup_env_client.sh" \
        "${HWGOAL_XPL_ROOT}/utils" "${HWGOAL_ADAPTER_DIR}/deploy.yml" "${eval_env_conda_env}" \
        "${policy_server_port}" "${bench_name}" "${task_name}" "${env_cfg_type}" HumanWAM_Goal \
        "${additional_info}" "${HWGOAL_BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
elif [[ "${eval_env_mode}" != "sim" ]]; then
    echo "[CLIENT][ERROR] HumanWAM-Goal supports EVAL_ENV_TYPE=sim or debug" >&2
    exit 2
fi

# Simulator: RoboTwin's pinned native evaluator with a fail-closed HumanWAM-Goal policy
# client; HWGOAL_SEED_MODE=fixed (default) replays the released cases, native re-selects.
goalwam_select_python "${eval_env_conda_env}"
exec env \
    CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    "${HWGOAL_EXEC_PYTHON}" "${HWGOAL_ADAPTER_DIR}/launch.py" client \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
        --port "${policy_server_port}" --host "${policy_server_ip}"
