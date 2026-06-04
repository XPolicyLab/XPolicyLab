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
policy_server_host=${11:-localhost}

export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"
policy_name="$(basename "${SCRIPT_DIR}")"

ckpt_run_id="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
legacy_run_id="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}-seed${seed}"
legacy_ckpt_run_id="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-seed${seed}"

is_valid_model_dir() {
    local dir="$1"
    [[ -d "${dir}" ]] || return 1
    [[ -f "${dir}/model.config.json" ]] && return 0
    [[ -f "${dir}/model.safetensors" ]] && return 0
    find "${dir}" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null | grep -q .
}

resolve_model_dir() {
    local candidate=""
    for candidate in \
        "${SCRIPT_DIR}/workspace/${ckpt_run_id}/model" \
        "${SCRIPT_DIR}/workspace/${legacy_ckpt_run_id}/model" \
        "${SCRIPT_DIR}/workspace/${legacy_run_id}/model"; do
        if is_valid_model_dir "${candidate}"; then
            echo "$(cd "${candidate}" && pwd)"
            return 0
        fi
    done
    return 1
}

model_dir="${HOLOBRAIN_MODEL_DIR:-}"
if [[ -n "${model_dir}" && "${model_dir}" != "null" ]]; then
    [[ "${model_dir}" = /* ]] || model_dir="${SCRIPT_DIR}/${model_dir}"
elif model_dir="$(resolve_model_dir)"; then
    :
else
    echo -e "\033[31m[SERVER] exported model not found for ckpt_run_id=${ckpt_run_id}\033[0m" >&2
    echo -e "\033[31m[SERVER] tried workspace/${ckpt_run_id}/model and legacy workspace/${legacy_run_id}/model\033[0m" >&2
    echo -e "\033[31m[SERVER] Run export.sh on the training workspace, or set HOLOBRAIN_MODEL_DIR / deploy.yml model_dir\033[0m" >&2
    exit 1
fi

if ! is_valid_model_dir "${model_dir}"; then
    echo -e "\033[31m[SERVER] invalid model_dir: ${model_dir}\033[0m" >&2
    exit 1
fi

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
inference_prefix="${HOLOBRAIN_INFERENCE_PREFIX:-robotwin2_0}"

echo -e "\033[33m[SERVER] policy=${policy_name} task=${task_name} ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] ckpt_run_id=${ckpt_run_id}\033[0m"
echo -e "\033[33m[SERVER] model_dir=${model_dir}\033[0m"
echo -e "\033[33m[SERVER] inference_prefix=${inference_prefix}\033[0m"
echo -e "\033[33m[SERVER] host=${policy_server_host} port=${policy_server_port} gpu=${policy_gpu_id}\033[0m"
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            policy_name="${policy_name}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            gpu_id="${policy_gpu_id}" \
            model_dir="${model_dir}" \
            inference_prefix="${inference_prefix}"
