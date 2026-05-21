#!/bin/bash
set -e
set -o pipefail

policy_name=GR00T_N1_6
dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
policy_gpu_id=${6}
seed=${7}
policy_conda_env=${8}
eval_env_conda_env=${9}
env_gpu_id=${10:-${policy_gpu_id}}
MODEL_PATH=${11:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"

export GR00T_XPOLICYLAB_POLICY_DIR="${SCRIPT_DIR}"
export GR00T_XPOLICYLAB_ALIAS="${policy_name}"
export PYTHONPATH="${SCRIPT_DIR}/xpolicylab_alias:${SCRIPT_DIR}/Isaac-GR00T:${ROOT_DIR}:${PYTHONPATH}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")
echo -e "\033[33m[INFO] Policy GPU ID: ${policy_gpu_id}\033[0m"
echo -e "\033[33m[INFO] Env GPU ID: ${env_gpu_id}\033[0m"
echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"

run_basename="${task_name}-gr00t-${action_type}-${expert_data_num}eps-seed${seed}"
latest_file="${SCRIPT_DIR}/checkpoints/${run_basename}.latest"

if [ -z "${MODEL_PATH}" ]; then
    if [ -f "${latest_file}" ]; then
        run_dir="$(cat "${latest_file}")"
    else
        run_dir=$(ls -dt "${SCRIPT_DIR}/checkpoints/${run_basename}"-* 2>/dev/null | head -1 || true)
    fi
    if [ -n "${run_dir:-}" ] && [ -d "${run_dir}" ]; then
        MODEL_PATH=$(ls -d "${run_dir}"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1 || true)
        if [ -z "${MODEL_PATH}" ] && [ -f "${run_dir}/config.json" ]; then
            MODEL_PATH="${run_dir}"
        fi
    fi
fi

if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then
    echo -e "\033[31m[ERROR] No checkpoint found. Pass MODEL_PATH as the 11th argument or run train.sh first.\033[0m"
    echo -e "\033[31m[ERROR] Checked latest marker: ${latest_file}\033[0m"
    exit 1
fi
echo -e "\033[33m[INFO] Model path: ${MODEL_PATH}\033[0m"

cleanup(){
    [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        env_cfg_type="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        model_path="${MODEL_PATH}" \
        device="cuda:0" \
    &
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID}\033[0m"

additional_info="model_path=${MODEL_PATH},action_type=${action_type}"
bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${FREE_PORT}" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${ROOT_DIR}" \
    "${seed}" \
    "${env_gpu_id}"
