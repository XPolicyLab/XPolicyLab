#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_env=${8:?policy environment name, Python path, or XBRAIN_PYTHON is required}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

PYTHONPATH_PARTS=("${BENCH_ROOT}")

resolve_python() {
    if [[ -n "${XBRAIN_PYTHON:-}" ]]; then
        [[ -x "${XBRAIN_PYTHON}" ]] || { echo "[SERVER][ERROR] XBRAIN_PYTHON is not executable: ${XBRAIN_PYTHON}" >&2; return 1; }
        realpath "${XBRAIN_PYTHON}"
        return
    fi
    if [[ "${policy_env}" == /* && -x "${policy_env}/bin/python" ]]; then
        realpath "${policy_env}/bin/python"
        return
    fi
    if [[ -x "${policy_env}" ]]; then
        realpath "${policy_env}"
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "${policy_env}"
        command -v python
        return
    fi
    echo "[SERVER][ERROR] Cannot resolve Python. Set XBRAIN_PYTHON=/path/to/python, pass an environment path, or install Conda." >&2
    return 1
}

PYTHON_BIN="$(resolve_python)"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
