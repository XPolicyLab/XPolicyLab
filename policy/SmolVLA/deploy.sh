#!/bin/bash
set -e

policy_name=SmolVLA
gpu_id=${1}
policy_conda_env=${2}
PRETRAINED_PATH=${3}
PORT=${4:-6000}
DEVICE=${5:-cuda}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${PORT}" \
        policy_name="${policy_name}" \
        pretrained_path="${PRETRAINED_PATH}" \
        device="${DEVICE}"
