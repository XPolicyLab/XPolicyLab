#!/bin/bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "Usage: bash train_10step_save.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id> [extra_args...]"
    echo "Example: bash train_10step_save.sh RoboDojo stack_bowls arx_x5 3500 joint 0 0,1,2,3,4,5,6,7"
    exit 1
fi

dataset_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}
shift 7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"
STARVLA_ADAPTER="${SCRIPT_DIR}/starvla_adapter"

config_yaml="${SCRIPT_DIR}/xpolicy_oft_vla.yaml"
data_dir_name="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
run_id="${data_dir_name}-${seed}-smoke10"
num_processes=$(awk -F',' '{print NF}' <<< "${gpu_id}")

echo "[starVLA] config_yaml=${config_yaml}"
echo "[starVLA] run_id=${run_id}"
echo "[starVLA] seed=${seed}"
echo "[starVLA] dataset is configured in xpolicy_oft_vla.yaml"
echo "[starVLA] train_entry=starVLA/training/train_starvla.py"
echo "[starVLA] max_train_steps=10, save_interval=10"
echo "[starVLA] num_processes=${num_processes}, mixed_precision=bf16"

cd "${STARVLA_ROOT}"
STARVLA_EXTRA_DATA_REGISTRY="${STARVLA_ADAPTER}/data_registry" \
PYTHONPATH="${STARVLA_ADAPTER}:${STARVLA_ROOT}:${PYTHONPATH:-}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch \
    --num_processes "${num_processes}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --run_id "${run_id}" \
    --seed "${seed}" \
    --trainer.max_train_steps 10 \
    --trainer.num_warmup_steps 0 \
    --trainer.save_interval 10 \
    --trainer.eval_interval 999999 \
    --trainer.logging_frequency 1 \
    "$@"
