#!/bin/bash
set -e

policy_name=InternVLA-A1
gpu_id=${1}
policy_conda_env=${2}
CKPT_PATH=${3}
PORT=${4:-6000}
DEVICE=${5:-cuda}
STATS_KEY=${6:-aloha}
DTYPE=${7:-float32}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/InternVLA-A1/deploy.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${PORT}" \
        policy_name="${policy_name}" \
        ckpt_path="${CKPT_PATH}" \
        stats_key="${STATS_KEY}" \
        dtype="${DTYPE}" \
        device="${DEVICE}"
