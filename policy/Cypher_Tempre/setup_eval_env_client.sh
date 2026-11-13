#!/usr/bin/env bash
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
ROBODOJO_EVAL_ROOT="${ROBODOJO_ROOT:-${BENCH_ROOT}/RoboDojo-eval}"
export ROBODOJO_ROOT="${ROBODOJO_EVAL_ROOT}"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="${XPL_POLICY_NAME:-$(basename "${SCRIPT_DIR}")}"
deploy_yml="${XPL_DEPLOY_YML:-${SCRIPT_DIR}/deploy.yml}"

# RoboDojo invokes this script from the adapter directory. Leave it before
# any helper that may start Python.
cd "${XPL_ROOT}"

require_client_deps() {
    local python_bin=$1
    local check_output
    if ! check_output=$("${python_bin}" -c \
        'import openai; assert openai.__version__ == "3.8.0"' 2>&1); then
        echo "[CLIENT][ERROR] Required openai==3.8.0 is missing or wrong version." >&2
        echo "${check_output}" >&2
        echo "[CLIENT][ERROR] Run: bash ${SCRIPT_DIR}/install.sh ${python_bin}" >&2
        exit 1
    fi
    if ! check_output=$("${python_bin}" -c \
        'from importlib.metadata import version; assert version("pillow") == "12.3.0"' 2>&1); then
        echo "[CLIENT][ERROR] Required pillow==12.3.0 is missing or wrong version." >&2
        echo "${check_output}" >&2
        echo "[CLIENT][ERROR] Run: bash ${SCRIPT_DIR}/install.sh ${python_bin}" >&2
        exit 1
    fi
}

resolve_client_python() {
    local eval_env=$1
    if [[ "${eval_env}" == "uv" ]]; then
        echo "${ROBODOJO_EVAL_ROOT}/.venv/bin/python"
        return 0
    fi
    if [[ -x "${eval_env}/bin/python" ]]; then
        echo "${eval_env}/bin/python"
        return 0
    fi
    if [[ -x "${eval_env}" ]]; then
        echo "${eval_env}"
        return 0
    fi
    if ! command -v conda >/dev/null 2>&1; then
        return 1
    fi
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${eval_env}"
    command -v python
    conda deactivate
}

if [[ "${EVAL_ENV_TYPE:-sim}" == "debug" ]]; then
    debug_python="$(resolve_client_python "${eval_env_conda_env}" || true)"
    if [[ -z "${debug_python}" || ! -x "${debug_python}" ]]; then
        echo "[CLIENT][ERROR] Could not resolve Python for eval env: ${eval_env_conda_env}" >&2
        exit 1
    fi
    require_client_deps "${debug_python}"
    export PYTHONPATH="${BENCH_ROOT}:${PYTHONPATH:-}"
    debug_client="${XPL_ROOT}/debug_env_client.py"
    if [[ ! -f "${debug_client}" ]]; then
        debug_client="${XPL_ROOT}/utils/debug_env_client.py"
    fi
    exec "${debug_python}" "${debug_client}" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "${policy_name}" \
        --protocol ws \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_episode_num 1 \
        --eval_batch false
fi

if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" ]]; then
    eval_env_path="${eval_env_conda_env}"
    if [[ "${eval_env_path}" == "uv" ]]; then
        eval_env_path="${ROBODOJO_EVAL_ROOT}/.venv"
    fi
    if [[ -x "${eval_env_path}/bin/python" ]]; then
        source "${eval_env_path}/bin/activate"
        require_client_deps "${eval_env_path}/bin/python"
        native_cuda="${ROBODOJO_EVAL_ROOT}/.cuda-native"
        if [[ -e "${native_cuda}/libcuda.so.1" ]]; then
            export LD_LIBRARY_PATH="${native_cuda}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
        export OMNI_KIT_ACCEPT_EULA=YES
        export PYTHONPATH="${BENCH_ROOT}:${ROBODOJO_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        exec bash "${ROBODOJO_EVAL_ROOT}/scripts/eval_policy.sh" \
            --root_dir "${ROBODOJO_EVAL_ROOT}" \
            --task_name "${task_name}" \
            --env_cfg_type "${env_cfg_type}" \
            --device_id "${env_gpu_id}" \
            --policy_name "${policy_name}" \
            --host "${policy_server_ip}" \
            --port "${policy_server_port}" \
            --protocol ws \
            --eval_batch false \
            --additional_info "${additional_info}" \
            --seed "${seed}"
    fi
fi

client_python="$(resolve_client_python "${eval_env_conda_env}" || true)"
if [[ -z "${client_python}" || ! -x "${client_python}" ]]; then
    echo "[CLIENT][ERROR] Could not resolve Python for eval env: ${eval_env_conda_env}" >&2
    exit 1
fi
require_client_deps "${client_python}"

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${deploy_yml}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${ROBODOJO_EVAL_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
