#!/bin/bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: bash setup_eval_policy_server.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_conda_env> <policy_server_port> [policy_server_host]"
    exit 1
fi

dataset_name=$1
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
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

dexora_root="${DEXORA_ROOT:-/root/crx/Dexora}"
checkpoint_path="${DEXORA_CKPT_PATH:-}"
if [[ -z "${checkpoint_path}" ]]; then
    if [[ "${ckpt_name}" = /* || -e "${ckpt_name}" ]]; then
        checkpoint_path="${ckpt_name}"
    else
        checkpoint_path="${dexora_root}/checkpoints/${ckpt_name}"
    fi
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, server=${policy_server_host}:${policy_server_port}"
echo "[SERVER] checkpoint_path=${checkpoint_path}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    HF_HOME="${HF_HOME:-/mnt/data/crx/hf_cache}" \
    HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/data/crx/hf_cache/hub}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    PYTHONPATH="${ROOT_DIR}:${dexora_root}:${PYTHONPATH:-}" \
    python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            checkpoint_path="${checkpoint_path}" \
            dexora_root="${dexora_root}" \
            config_path="${DEXORA_CONFIG_PATH:-${dexora_root}/configs/base.yaml}" \
            text_encoder_path="${DEXORA_T5:-google/t5-v1_1-xxl}" \
            vision_encoder_path="${DEXORA_SIGLIP:-google/siglip-so400m-patch14-384}" \
            hf_home="${HF_HOME:-/mnt/data/crx/hf_cache}" \
            hf_hub_cache="${HF_HUB_CACHE:-/mnt/data/crx/hf_cache/hub}" \
            hf_offline=True \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
