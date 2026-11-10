#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
goalwam_select_python "${policy_conda_env}"

echo "[SERVER] policy=HumanWAM-Goal, task=${task_name}, policy_server_port=${policy_server_port}"

# launch.py resolves release paths, then execs XPolicyLab's setup_policy_server.py.
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${HWGOAL_EXEC_PYTHON}" "${HWGOAL_ADAPTER_DIR}/launch.py" server \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" \
        --port "${policy_server_port}" --host "${policy_server_host}"
