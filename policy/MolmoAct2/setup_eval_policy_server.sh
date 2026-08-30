#!/bin/bash
set -euo pipefail
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"

if [[ "${policy_conda_env}" == "uv" || "${policy_conda_env}" == */* ]]; then
    if [[ "${policy_conda_env}" == "uv" ]]; then
        resolver_python="${SCRIPT_DIR}/.venv/bin/python"
        if [[ ! -x "${resolver_python}" ]]; then
            echo "[SERVER][ERROR] uv venv not found: ${SCRIPT_DIR}/.venv" >&2
            echo "[SERVER][ERROR] Run: bash ${SCRIPT_DIR}/install.sh" >&2
            exit 1
        fi
        policy_uv_env_path=$("${resolver_python}" - <<PYENV
import yaml
from pathlib import Path
script_dir = Path("${SCRIPT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
path = Path(cfg["policy_uv_env_path"]).expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
)
    else
        policy_uv_env_path="${policy_conda_env}"
        if [[ "${policy_uv_env_path}" != /* ]]; then
            policy_uv_env_path="${SCRIPT_DIR}/${policy_uv_env_path}"
        fi
        policy_uv_env_path="$(cd "${policy_uv_env_path}" && pwd)"
    fi
    if [[ ! -f "${policy_uv_env_path}/.venv/bin/activate" ]]; then
        echo "[SERVER][ERROR] uv venv not found: ${policy_uv_env_path}/.venv" >&2
        echo "[SERVER][ERROR] Run: bash ${SCRIPT_DIR}/install.sh" >&2
        exit 1
    fi
    echo "[SERVER] Activating uv environment: ${policy_uv_env_path}/.venv"
    source "${policy_uv_env_path}/.venv/bin/activate"
    PYTHON_BIN="$(command -v python)"
else
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if type deactivate >/dev/null 2>&1 && [[ -n "${VIRTUAL_ENV:-}" ]]; then
        deactivate || true
    fi
    unset VIRTUAL_ENV
    echo "[SERVER] Activating Conda environment: ${policy_conda_env}"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
echo "[SERVER] Using python: ${PYTHON_BIN}"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
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
            action_type="${action_type}" \
            action_dim="${action_dim}"
