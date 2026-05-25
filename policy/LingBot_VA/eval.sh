#!/bin/bash
set -e

policy_name="$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
dataset_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
expert_data_num=${5}
action_type=${6}
seed=${7}
policy_gpu_id=${8}
env_gpu_id=${9}
policy_conda_env=${10}
eval_env_conda_env=${11}
CONFIG_NAME=${12:-robotwin30_train}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

if [[ -z "${dataset_name}" || -z "${task_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${expert_data_num}" || -z "${action_type}" || -z "${seed}" || -z "${policy_gpu_id}" || -z "${env_gpu_id}" || -z "${policy_conda_env}" || -z "${eval_env_conda_env}" ]]; then
    echo "Usage: bash eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env> [config_name]"
    exit 1
fi

if [[ "${ckpt_name}" = /* ]]; then
    CHECKPOINT_PATH="${ckpt_name}"
else
    CHECKPOINT_PATH="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/checkpoints/${ckpt_name}"
fi

echo -e "\033[33m[INFO] Policy GPU ID: ${policy_gpu_id}\033[0m"
echo -e "\033[33m[INFO] Env GPU ID: ${env_gpu_id}\033[0m"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${FREE_PORT}"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        ckpt_name="${ckpt_name}" \
        env_cfg_type="${env_cfg_type}" \
        env_cfg="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        checkpoint_path="${CHECKPOINT_PATH}" \
        config_name="${CONFIG_NAME}" \
    &

SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
bash "${UTILS_DIR}/setup_env_client.sh" "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" "${FREE_PORT}" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" "${additional_info}" "${ROOT_DIR}" "${seed}" "${env_gpu_id}"
echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"