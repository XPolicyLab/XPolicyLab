#!/bin/bash
set -e

dataset_name=${1:-"RoboDojo"}
task_name=${2:-"stack_bowls"}
ckpt_name=${3:-"Pi_05_sim_arx-x5_seed_1"}
env_cfg_type=${4:-"arx_x5"}
expert_data_num=${5:-100}
action_type=${6:-"joint"}
seed=${7:-0}
policy_gpu_id=${8:-1}
policy_conda_env=${9:-"uv"}
policy_server_port=${10:-0}
policy_server_host=${11:-"localhost"}
protocol=${12:-robodojo_ws}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${CURRENT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

if [[ -z "${policy_server_port}" || "${policy_server_port}" == "0" ]]; then
    policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
fi

resolve_uv_env_path() {
    local raw_path=$1
    if [[ "${raw_path}" == "uv" ]]; then
        python3 - <<PYENV
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
        python3 - <<PYENV
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
path = Path("${raw_path}").expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
    fi
}

if [[ "${policy_conda_env}" == "uv" || "${policy_conda_env}" == */* ]]; then
    policy_uv_env_path="$(resolve_uv_env_path "${policy_conda_env}")"
    PYTHON_BIN="${policy_uv_env_path}/.venv/bin/python"
    echo "[SERVER] Using uv environment: ${policy_uv_env_path}"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "[SERVER] Activating Conda environment: ${policy_conda_env}"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[SERVER] Python not found: ${PYTHON_BIN}" >&2
    echo "[SERVER] Run: cd ${CURRENT_DIR}/openpi && UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot" >&2
    exit 1
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] Using python: ${PYTHON_BIN}"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            policy_server_port="${policy_server_port}" \
            policy_server_host="${policy_server_host}" \
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
            action_dim="${action_dim}" \
            protocol="${protocol}"
