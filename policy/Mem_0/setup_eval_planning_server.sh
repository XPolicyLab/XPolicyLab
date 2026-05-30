#!/bin/bash
set -euo pipefail

# Start vLLM for Mem_0 Mn planning module (merged Qwen3-VL-8B weights).
#
# Args:
#   dataset_name ckpt_name env_cfg_type expert_data_num action_type seed
#   planning_gpu_ids planning_port [policy_dir]

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
planning_gpu_ids=$7
planning_port=$8
policy_dir=${9:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${policy_dir:-${SCRIPT_DIR}}"
UPSTREAM_DIR="${POLICY_DIR}/Mem_0"

run_name="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-seed${seed}"
merged_dir="${MEM0_PLANNING_MERGED_PATH:-${UPSTREAM_DIR}/checkpoints/${run_name}_planning_merged}"

if [[ ! -d "${merged_dir}" ]]; then
    echo -e "\033[31m[PLANNING] merged weights not found: ${merged_dir}\033[0m" >&2
    echo "Train planning first: bash train.sh ... planning  (or set MEM0_PLANNING_MERGED_PATH)" >&2
    exit 1
fi

tp_size="${MEM0_VLLM_TP_SIZE:-1}"
if [[ "${planning_gpu_ids}" == *","* ]]; then
    IFS=',' read -ra _gpus <<< "${planning_gpu_ids}"
    tp_size="${MEM0_VLLM_TP_SIZE:-${#_gpus[@]}}"
fi

conda_env="${CONDA_ENV_VLLM:-vllm}"
echo -e "\033[33m[PLANNING] merged=${merged_dir}\033[0m"
echo -e "\033[33m[PLANNING] GPUs=${planning_gpu_ids} port=${planning_port} tp=${tp_size}\033[0m"
echo -e "\033[33m[PLANNING] VLLM_URL=http://127.0.0.1:${planning_port}/v1\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${conda_env}"

exec env \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${planning_gpu_ids}" \
    vllm serve "${merged_dir}" \
        --tensor-parallel-size "${tp_size}" \
        --mm-encoder-tp-mode data \
        --host 0.0.0.0 \
        --port "${planning_port}"
