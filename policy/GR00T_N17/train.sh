#!/usr/bin/env bash
# GLOBAL_BATCH_SIZE=640 MAX_STEPS=60000 bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7

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
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
DATA_ROOT="${GR00T_LEROBOT_HOME:-}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Set GR00T_LEROBOT_HOME to the LeRobot datasets root." >&2
  exit 1
fi

base_model="${GR00T_BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
cosmos_model="${GR00T_COSMOS_MODEL:-nvidia/Cosmos-Reason2-2B}"

data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
dataset_path="${DATA_ROOT}/${data_setting}"
modality_config="${POLICY_DIR}/configs/${env_cfg_type}_config.py"
output_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"
export GR00T_COSMOS_MODEL="${cosmos_model}"
export GR00T_VIDEO_BACKEND="${GR00T_VIDEO_BACKEND:-pyav}"

if [[ ! -d "${dataset_path}" ]]; then
  echo "Processed dataset not found: ${dataset_path}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${modality_config}" ]]; then
  echo "Modality config not found: ${modality_config}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ -f "${base_model}/config.json" ]]; then
  :
elif [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "GR00T base model not found locally: ${base_model}" >&2
  echo "Unset HF_HUB_OFFLINE or set GR00T_BASE_MODEL to a local directory." >&2
  exit 1
fi

if [[ -f "${cosmos_model}/config.json" ]]; then
  :
elif [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "Cosmos backbone not found locally: ${cosmos_model}" >&2
  echo "Unset HF_HUB_OFFLINE or set GR00T_COSMOS_MODEL to a local directory." >&2
  exit 1
fi

mkdir -p "${output_dir}"

MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-640}"
USE_WANDB="${USE_WANDB:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

export MAX_STEPS SAVE_STEPS GLOBAL_BATCH_SIZE USE_WANDB DATALOADER_NUM_WORKERS

echo "[GR00T_N17] dataset_path=${dataset_path}"
echo "[GR00T_N17] base_model=${base_model}"
echo "[GR00T_N17] cosmos_model=${cosmos_model}"
echo "[GR00T_N17] video_backend=${GR00T_VIDEO_BACKEND}"
echo "[GR00T_N17] output_dir=${output_dir}"
echo "[GR00T_N17] num_gpus=${NUM_GPUS}"
echo "[GR00T_N17] global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "[GR00T_N17] per_gpu_batch_size=$((GLOBAL_BATCH_SIZE / NUM_GPUS))"
echo "[GR00T_N17] max_steps=${MAX_STEPS}"
echo "[GR00T_N17] save_steps=${SAVE_STEPS}"

cd "${GR00T_ROOT}"
source .venv/bin/activate

uv run --no-sync bash examples/finetune.sh \
  --base-model-path "${base_model}" \
  --dataset-path "${dataset_path}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${modality_config}" \
  --output-dir "${output_dir}" \
  --experiment-name "${ckpt_setting}"
