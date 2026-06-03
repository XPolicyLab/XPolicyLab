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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GR00T_ROOT="${SCRIPT_DIR}/gr00t_n17"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"

resolve_uv_env() {
    local raw_path=$1
    if [[ "${raw_path}" == "uv" ]]; then
        python - <<PYENV
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
path = Path(cfg.get("policy_uv_env_path", "gr00t_n17")).expanduser()
if not path.is_absolute():
    path = (Path("${SCRIPT_DIR}") / path).resolve()
print(path)
PYENV
    else
        python - <<PYENV
from pathlib import Path
path = Path("${raw_path}").expanduser()
if not path.is_absolute():
    path = (Path("${SCRIPT_DIR}") / path).resolve()
print(path)
PYENV
    fi
}

if [[ "${policy_conda_env}" == "uv" || "${policy_conda_env}" == */* ]]; then
    policy_uv_env_path="$(resolve_uv_env "${policy_conda_env}")"
    PYTHON_BIN="${policy_uv_env_path}/.venv/bin/python"
    echo "[SERVER] Using uv environment: ${policy_uv_env_path}"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "[SERVER] Activating Conda environment: ${policy_conda_env}"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="$(command -v python)"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONPATH="${GR00T_ROOT}:${PYTHONPATH:-}"
# Allow HuggingFace download for cosmos_model_path; set HF_HUB_OFFLINE=1 for fully offline deploy
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export GR00T_VIDEO_BACKEND="${GR00T_VIDEO_BACKEND:-pyav}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
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
