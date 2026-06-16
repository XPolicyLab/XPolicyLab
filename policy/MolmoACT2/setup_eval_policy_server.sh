#!/bin/bash
set -e

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_conda_env=$9
policy_server_port=${10}
policy_server_host=${11:-"localhost"}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${CURRENT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"

# HuggingFace hub download may require proxy on deploy hosts (proxyup alias).
_DEPLOY_PROXY_HOST="${DEPLOY_PROXY_HOST:-192.168.16.76}"
_DEPLOY_PROXY_PORT="${DEPLOY_PROXY_PORT:-18000}"
export http_proxy="http://${_DEPLOY_PROXY_HOST}:${_DEPLOY_PROXY_PORT}"
export https_proxy="${http_proxy}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
echo "[SERVER] http_proxy=${http_proxy}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
YAML_PYTHON="${CONDA_BASE}/bin/python"
if type deactivate >/dev/null 2>&1 && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate || true
fi
unset VIRTUAL_ENV
if [[ "${policy_conda_env}" == "uv" || "${policy_conda_env}" == */* ]]; then
    if [[ "${policy_conda_env}" == "uv" ]]; then
        policy_uv_env_path=$("${YAML_PYTHON}" - <<PYENV
import yaml
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
path = Path(cfg["policy_uv_env_path"]).expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
)
    else
        policy_uv_env_path=$("${YAML_PYTHON}" - <<PYENV
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
path = Path("${policy_conda_env}").expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
)
    fi
    echo "[SERVER] Activating uv environment: ${policy_uv_env_path}/.venv"
    source "${policy_uv_env_path}/.venv/bin/activate"
    PYTHON_BIN="$(command -v python)"
else
    echo "[SERVER] Activating Conda environment: ${policy_conda_env}"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
echo "[SERVER] Using python: ${PYTHON_BIN}"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"
