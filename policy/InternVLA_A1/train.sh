#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
repo_id="${INTERNVLA_REPO_ID:-${data_setting}}"
intern_action_mode="${INTERNVLA_ACTION_MODE:-delta}"
use_external_stats="${INTERNVLA_USE_EXTERNAL_STATS:-true}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_HOME="${HF_HOME:-/xspark-cache/shared}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/xspark-cache/shared/lerobot}"
export COSMOS_PATH="${COSMOS_PATH:-/mnt/xspark-data/xspark_shared/model_weights/Cosmos-Tokenizer-CI8x8}"
export QWEN3_2B_PATH="${QWEN3_2B_PATH:-/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-2B-Instruct}"
export PROC_PER_NODE="${PROC_PER_NODE:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"
export JOB_NAME="${ckpt_setting}"
export OUTPUT_DIR="${ckpt_dir}"
export TRAIN_SEED="${seed}"

echo "[InternVLA_A1] repo_id=${repo_id}"
echo "[InternVLA_A1] checkpoint_dir=${ckpt_dir}"

bash "${POLICY_DIR}/internvla_a1/launch/internvla_a1_3b_finetune.sh" \
  "${repo_id}" \
  "${intern_action_mode}" \
  "${use_external_stats}" \
  "${ckpt_dir}" \
  "${seed}"