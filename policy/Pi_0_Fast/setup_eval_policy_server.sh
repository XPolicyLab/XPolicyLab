#!/bin/bash
set -e
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_uv_env=${9:-uv}
policy_server_port=${10}
policy_server_host=${11:-"localhost"}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${CURRENT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
YAML_PYTHON="${CONDA_BASE}/bin/python"

resolve_uv_env() {
    local raw_path=$1
    if [[ "${raw_path}" == "uv" ]]; then
        "${YAML_PYTHON}" - <<PYENV
import yaml
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
path = Path(cfg["policy_uv_env_path"]).expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
    else
        "${YAML_PYTHON}" - <<PYENV
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
path = Path("${raw_path}").expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
    fi
}

policy_uv_env_path="$(resolve_uv_env "${policy_uv_env}")"
if [[ ! -f "${policy_uv_env_path}/.venv/bin/activate" ]]; then
    echo "[SERVER][ERROR] uv venv not found: ${policy_uv_env_path}/.venv" >&2
    echo "[SERVER][ERROR] Run: bash ${CURRENT_DIR}/install.sh" >&2
    exit 1
fi

echo "[SERVER] Activating uv environment: ${policy_uv_env_path}/.venv"
source "${policy_uv_env_path}/.venv/bin/activate"
PYTHON_BIN="$(command -v python)"
OPENPI_SRC="${policy_uv_env_path}/src"
echo "[SERVER] Using python: ${PYTHON_BIN}"

PYTHONPATH_PARTS=("${ROOT_DIR}")
if [[ -d "${OPENPI_SRC}" ]]; then
    PYTHONPATH_PARTS+=("${OPENPI_SRC}")
fi

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"
