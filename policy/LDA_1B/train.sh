#!/bin/bash
# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
expert_data_num=${4:?expert_data_num required}
action_type=${5:?action_type required}
seed=${6:?seed required}
gpu_id=${7:?gpu_id required}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/LDA-1B"
ADAPTER_DIR="${UPSTREAM_DIR}/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"

ckpt_root_dir="${LDA_CKPT_ROOT:-${SCRIPT_DIR}/checkpoints}"
data_root_dir="${LDA_DATA_ROOT:-${SCRIPT_DIR}/data}"

# Training knobs. Keep robot/model shape contracts in the YAML; tune run-scale
# hyperparameters here, matching the original LDA run scripts style.
num_processes="${LDA_NUM_PROCESSES:-8}"
per_device_batch_size="${LDA_PER_DEVICE_BATCH_SIZE:-16}"
max_train_steps="${LDA_MAX_TRAIN_STEPS:-50000}"
save_interval="${LDA_SAVE_INTERVAL:-5000}"
eval_interval="${LDA_EVAL_INTERVAL:-1000}"
logging_frequency="${LDA_LOGGING_FREQUENCY:-1000}"
learning_rate="${LDA_LEARNING_RATE:-4e-5}"
repeated_diffusion_steps="${LDA_REPEATED_DIFFUSION_STEPS:-1}"
freeze_module_list="${LDA_FREEZE_MODULES:-}"
training_task_weights="${LDA_TRAINING_TASK_WEIGHTS:-[1,1,1,1]}"
wandb_project="${LDA_WANDB_PROJECT:-lda}"
wandb_entity="${LDA_WANDB_ENTITY:-}"
is_debug="${LDA_DEBUG:-False}"

dataset_id="${LDA_DATASET_ID:-$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")}"
resolved_dataset_dir="$(xpolicylab_resolve_dataset_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
  "${env_cfg_type}" "${action_type}" "${expert_data_num}")"
if [[ ! -d "${resolved_dataset_dir}" ]]; then
  echo -e "\033[31m[train.sh] dataset not found: ${resolved_dataset_dir}\033[0m" >&2
  echo -e "\033[31m           Run process_data.sh first (expected tag: ${dataset_id}).\033[0m" >&2
  exit 1
fi
dataset_id="$(basename "${resolved_dataset_dir}")"

export XPOLICYLAB_DATASET_ID="${XPOLICYLAB_DATASET_ID:-${dataset_id}}"
export XPOLICYLAB_ROBOT_TYPE="${XPOLICYLAB_ROBOT_TYPE:-${env_cfg_type}}"
ckpt_setting="${LDA_CKPT_SETTING:-$(xpolicylab_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")}"

default_pretrained_ckpt="${SCRIPT_DIR}/checkpoints/LDA-pretrain/LDA-pretrain.pt"
if [[ -n "${LDA_PRETRAINED_CHECKPOINT:-}" ]]; then
    pretrained_checkpoint="${LDA_PRETRAINED_CHECKPOINT}"
elif [[ -f "${default_pretrained_ckpt}" ]]; then
    pretrained_checkpoint="${default_pretrained_ckpt}"
    echo -e "\033[33m[train.sh] Loading LDA-1B pretrain ckpt: ${pretrained_checkpoint}\033[0m"
else
    pretrained_checkpoint="null"
    echo -e "\033[31m[train.sh] WARNING: No pretrained checkpoint found at ${default_pretrained_ckpt}.\033[0m"
    echo -e "\033[31m            Training a 1B model from scratch on this dataset will likely\033[0m"
    echo -e "\033[31m            collapse to mean output on harder tasks. Download LDA-pretrain.pt\033[0m"
    echo -e "\033[31m            from https://huggingface.co/Wayer2/LDA-pretrain or pass\033[0m"
    echo -e "\033[31m            LDA_PRETRAINED_CHECKPOINT=<path> to silence this warning.\033[0m"
fi
mkdir -p "${ckpt_root_dir}/${ckpt_setting}"

echo -e "\033[33m[train] dataset_id=${XPOLICYLAB_DATASET_ID} ckpt_setting=${ckpt_setting}\033[0m"

cd "${UPSTREAM_DIR}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

training_cfg="${LDA_TRAINING_CONFIG:-lda/config/training/xpolicylab_arx_x5_LDA.yaml}"
accelerate_cfg="${LDA_ACCELERATE_CONFIG:-lda/config/deepseeds/deepspeed_zero2.yaml}"

accelerate launch \
  --config_file "${accelerate_cfg}" \
  --num_processes "${num_processes}" \
  lda/training/train_LDA.py \
  --config_yaml "${training_cfg}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.training_task_weights "${training_task_weights}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.eval_interval "${eval_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.learning_rate.base "${learning_rate}" \
  --trainer.repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --run_root_dir "${ckpt_root_dir}" \
  --run_id "${ckpt_setting}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  --is_debug "${is_debug}" \
  --seed "${seed}"
