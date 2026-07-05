#!/bin/bash
set -euo pipefail

# Start vLLM for Mem_0 Mn planning module (merged Qwen3-VL-8B weights).
#
# Args:
#   bench_name ckpt_name env_cfg_type action_type seed
#   planning_gpu_ids planning_port [policy_dir]

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
planning_gpu_ids=$6
planning_port=$7
policy_dir=${8:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${policy_dir:-${SCRIPT_DIR}}"
UPSTREAM_DIR="${POLICY_DIR}/Mem_0"
ADAPTER_DIR="${UPSTREAM_DIR}/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"

expert_data_num="${MEM0_EXPERT_DATA_NUM:-}"
merged_dir="${MEM0_PLANNING_MERGED_PATH:-$(mem0_resolve_planning_merged_dir "${POLICY_DIR}" \
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}")}"

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
