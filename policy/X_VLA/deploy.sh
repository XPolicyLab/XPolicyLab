#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <gpu_id> <policy_conda_env> <model_path> [processor_path] [port] [device]" >&2
    exit 1
fi

policy_name=X_VLA
gpu_id=${1}
policy_conda_env=${2}
MODEL_PATH=${3}
PROCESSOR_PATH=${4:-${MODEL_PATH}}
PORT=${5:-6000}
DEVICE=${6:-cuda}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/X_VLA/deploy.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${PORT}" \
        policy_name="${policy_name}" \
        model_path="${MODEL_PATH}" \
        processor_path="${PROCESSOR_PATH}" \
        device="${DEVICE}"
