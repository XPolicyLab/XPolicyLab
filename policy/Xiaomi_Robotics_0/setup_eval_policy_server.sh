#!/bin/bash
set -e

bench_name=$1
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
XR0_ROOT="${SCRIPT_DIR}/xiaomi_robotics_0/xr0"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"

_DEPLOY_PROXY_HOST="${DEPLOY_PROXY_HOST:-192.168.16.76}"
_DEPLOY_PROXY_PORT="${DEPLOY_PROXY_PORT:-18000}"
export http_proxy="http://${_DEPLOY_PROXY_HOST}:${_DEPLOY_PROXY_PORT}"
export https_proxy="${http_proxy}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
echo "[SERVER] http_proxy=${http_proxy}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export PYTHONPATH="${XR0_ROOT}:${PYTHONPATH:-}"
# Allow HuggingFace download for vlm_processor_path; set HF_HUB_OFFLINE=1 for fully offline deploy
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
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
