#!/bin/bash
set -euo pipefail

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

export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"
policy_name="$(basename "${SCRIPT_DIR}")"

ckpt_run_id="${BEINGH_CKPT_RUN_ID:-${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}}"

_resolve_latest_step() {
    local root="$1"
    if [[ -f "${root}/config.json" ]]; then
        echo "${root}"
        return 0
    fi
    local latest=""
    local step_dir
    for step_dir in "${root}"/*/; do
        [[ -d "${step_dir}" ]] || continue
        local base
        base="$(basename "${step_dir%/}")"
        if [[ "${base}" =~ ^[0-9]+$ ]] && [[ -f "${step_dir}/config.json" ]]; then
            latest="${step_dir%/}"
        fi
    done
    if [[ -n "${latest}" ]]; then
        echo "${latest}"
    else
        echo "${root}"
    fi
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    model_path="${MODEL_PATH}"
elif [[ -d "${ckpt_name}" && "${ckpt_name}" == */* ]]; then
    model_path="${ckpt_name}"
elif [[ -d "${SCRIPT_DIR}/checkpoints/${ckpt_run_id}" ]]; then
    model_path="$(_resolve_latest_step "${SCRIPT_DIR}/checkpoints/${ckpt_run_id}")"
else
    echo -e "\033[31m[SERVER] checkpoint not found: checkpoints/${ckpt_run_id}\033[0m" >&2
    exit 1
fi
model_path="$(cd "${model_path}" && pwd)"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] ckpt_run_id=${ckpt_run_id}\033[0m"
echo -e "\033[33m[SERVER] model_path=${model_path}\033[0m"
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export PYTHONPATH="${SCRIPT_DIR}/Being-H:${ROOT_DIR}:${PYTHONPATH:-}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="localhost" \
            policy_name="${policy_name}" \
            task_name="${task_name}" \
            data_project_name="${dataset_name}" \
            dataset_name="robodojo_posttrain" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            model_path="${model_path}"
