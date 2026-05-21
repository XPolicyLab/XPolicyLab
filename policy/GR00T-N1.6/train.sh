#!/bin/bash
set -e
set -o pipefail

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
gpu_id=${6}
seed=${7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
GR00T_DIR="${SCRIPT_DIR}/Isaac-GR00T"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${ROOT_DIR}:${GR00T_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/tmp}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${TMPDIR}" "${SCRIPT_DIR}/checkpoints"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"
echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"

dataset_id="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
dataset_path="${SCRIPT_DIR}/data/${dataset_id}"
if [ ! -d "${dataset_path}" ]; then
    echo -e "\033[33m[INFO] Converting XPolicyLab HDF5 data to GR00T LeRobot v2...\033[0m"
    bash "${SCRIPT_DIR}/process_data.sh" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" 30 "${SCRIPT_DIR}/data"
else
    echo -e "\033[33m[INFO] Using existing dataset: ${dataset_path}\033[0m"
fi

modality_config_path="${dataset_path}/xpolicylab_gr00t_config.py"
if [ ! -f "${modality_config_path}" ]; then
    echo -e "\033[31m[ERROR] Missing modality config: ${modality_config_path}\033[0m"
    exit 1
fi

run_basename="${task_name}-gr00t-${action_type}-${expert_data_num}eps-seed${seed}"
run_timestamp="${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
run_name="${RUNNAME:-${run_basename}-${run_timestamp}}"
output_root="${SCRIPT_DIR}/checkpoints"
run_dir="${output_root}/${run_name}"
echo "${run_dir}" > "${SCRIPT_DIR}/checkpoints/${run_basename}.latest"

base_model_path="${BASE_MODEL_PATH:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)/models/GR00T-N1.6-3B}"
if [ ! -d "${base_model_path}" ]; then
    echo -e "\033[31m[ERROR] Base model path not found: ${base_model_path}\033[0m"
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "${gpu_id}"
num_gpus=${#GPU_ARRAY[@]}
master_port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

export WANDB_PROJECT="${WANDB_PROJECT:-gr00t-xpolicylab}"
use_wandb_flag=()
if [ "${USE_WANDB:-0}" = "1" ]; then
    use_wandb_flag=(--use-wandb)
fi

save_steps="${SAVE_STEPS:-1000}"
max_steps="${MAX_STEPS:-10000}"
global_batch_size="${GLOBAL_BATCH_SIZE:-32}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"
episode_sampling_rate="${EPISODE_SAMPLING_RATE:-0.1}"
num_shards_per_epoch="${NUM_SHARDS_PER_EPOCH:-100000}"
shard_size="${SHARD_SIZE:-1024}"

echo -e "\033[33m[INFO] Starting GR00T finetuning: ${run_name}\033[0m"
cd "${GR00T_DIR}"

if [ "${num_gpus}" = "1" ]; then
    python "${SCRIPT_DIR}/finetune_xpolicylab.py" \
        --base-model-path "${base_model_path}" \
        --dataset-path "${dataset_path}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${modality_config_path}" \
        --output-dir "${output_root}" \
        --experiment-name "${run_name}" \
        --seed "${seed}" \
        --num-gpus "${num_gpus}" \
        --save-steps "${save_steps}" \
        --max-steps "${max_steps}" \
        --global-batch-size "${global_batch_size}" \
        --dataloader-num-workers "${dataloader_num_workers}" \
        --shard-size "${shard_size}" \
        --num-shards-per-epoch "${num_shards_per_epoch}" \
        --episode-sampling-rate "${episode_sampling_rate}" \
        --wandb-project "${WANDB_PROJECT}" \
        "${use_wandb_flag[@]}" \
        --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
else
    torchrun --nproc_per_node="${num_gpus}" --master_port="${master_port}" \
        "${SCRIPT_DIR}/finetune_xpolicylab.py" \
        --base-model-path "${base_model_path}" \
        --dataset-path "${dataset_path}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${modality_config_path}" \
        --output-dir "${output_root}" \
        --experiment-name "${run_name}" \
        --seed "${seed}" \
        --num-gpus "${num_gpus}" \
        --save-steps "${save_steps}" \
        --max-steps "${max_steps}" \
        --global-batch-size "${global_batch_size}" \
        --dataloader-num-workers "${dataloader_num_workers}" \
        --shard-size "${shard_size}" \
        --num-shards-per-epoch "${num_shards_per_epoch}" \
        --episode-sampling-rate "${episode_sampling_rate}" \
        --wandb-project "${WANDB_PROJECT}" \
        "${use_wandb_flag[@]}" \
        --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
fi

echo -e "\033[33m[INFO] Training complete. Run dir: ${run_dir}\033[0m"
