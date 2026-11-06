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

# In simulator mode WSA uses a small runtime wrapper around RoboTwin's official
# Python entry point. The official source remains untouched: the wrapper changes
# only the in-memory AST so --test-num is governed by the result counter and
# transient renderer-buffer failures cannot shorten a formal run.
source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)" || exit 1
if [[ "${eval_env_mode}" == "sim" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate || true
    conda activate "${eval_env_conda_env}"

    cd "${BENCH_ROOT}"
    robotwin_eval_args=()
    if [[ -n "${ROBOTWIN_EVAL_ARGS_FILE:-}" ]]; then
        if [[ ! -f "${ROBOTWIN_EVAL_ARGS_FILE}" ]]; then
            echo "[CLIENT][ERROR] Eval args file does not exist: ${ROBOTWIN_EVAL_ARGS_FILE}" >&2
            exit 1
        fi
        mapfile -t robotwin_eval_args < "${ROBOTWIN_EVAL_ARGS_FILE}"
    fi

    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${BENCH_ROOT}/XPolicyLab:${BENCH_ROOT}:${PYTHONPATH:-}" \
    python -u "${SCRIPT_DIR}/robotwin_eval_wrapper.py" \
        "${BENCH_ROOT}/scripts/eval_policy_xpolicylab.py" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "${policy_name}" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --protocol ws \
        --eval_batch false \
        --root_dir "${BENCH_ROOT}" \
        --device_id "${env_gpu_id}" \
        --additional_info "${additional_info}" \
        --seed "${seed}" \
        "${robotwin_eval_args[@]}"
    exit $?
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
