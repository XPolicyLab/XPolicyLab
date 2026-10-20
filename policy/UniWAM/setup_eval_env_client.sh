#!/bin/bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
env_gpu_id=${7:?}
eval_env_conda_env=${8:?}
additional_info=${9:-}
policy_server_port=${10:?}
policy_server_ip=${11:-127.0.0.1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

if [[ "${EVAL_ENV_TYPE:-sim}" == "debug" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${eval_env_conda_env}"
    export PYTHONPATH="${XPL_ROOT}:${WORKSPACE_ROOT}:${PYTHONPATH:-}"
    python "${XPL_ROOT}/utils/debug_env_client.py" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name UniWAM \
        --protocol ws \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_batch false \
        --eval_episode_num "${UNIWAM_DEBUG_EPISODES:-1}"
    exit 0
fi

if [[ "${bench_name}" != "RoboTwin" ]]; then
    echo "[ERROR] Simulator bridge supports bench_name=RoboTwin; got ${bench_name}." >&2
    exit 1
fi

robotwin_root=${UNIWAM_ROBOTWIN_ROOT:-${WORKSPACE_ROOT}/RoboTwin}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${eval_env_conda_env}"

client_args=(
    python "${SCRIPT_DIR}/robotwin_eval_client.py"
    --robotwin-root "${robotwin_root}"
    --host "${policy_server_ip}"
    --port "${policy_server_port}"
    --task-name "${task_name}"
    --task-config "${UNIWAM_ROBOTWIN_TASK_CONFIG:-demo_clean}"
    --seed "${seed}"
    --test-num "${UNIWAM_TEST_NUM:-1}"
    --result-path-tag "${UNIWAM_RESULT_PATH_TAG:-xpolicylab_uniwam}"
    --eval-video-log "${UNIWAM_EVAL_VIDEO_LOG:-true}"
    --checkpoint-root "${ckpt_name}"
)
if [[ -n "${UNIWAM_ELEMENT_SUMMARY_PATH:-}" ]]; then
    client_args+=(--summary-path "${UNIWAM_ELEMENT_SUMMARY_PATH}")
fi

exec env \
    CUDA_VISIBLE_DEVICES="${env_gpu_id}" \
    PYTHONPATH="${WORKSPACE_ROOT}:${XPL_ROOT}:${robotwin_root}:${PYTHONPATH:-}" \
    "${client_args[@]}"
