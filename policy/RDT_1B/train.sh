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
OUTPUT_DIR="${POLICY_DIR}/checkpoints/${ckpt_setting}"
NUM_GPUS="$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-google/t5-v1_1-xxl}"
export VISION_ENCODER_NAME="${VISION_ENCODER_NAME:-google/siglip-so400m-patch14-384}"
export RDT_HDF5_DIR="${RDT_HDF5_DIR:-${POLICY_DIR}/data/${data_setting}}"
export RDT_DATASET_NAME="${RDT_DATASET_NAME:-${data_setting}}"
export CFLAGS="${CFLAGS:--I/usr/include}"
export LDFLAGS="${LDFLAGS:--L/usr/lib/x86_64-linux-gnu}"
export CUTLASS_PATH="${CUTLASS_PATH:-/path/to/cutlass}"
export WANDB_PROJECT="${WANDB_PROJECT:-robotics_diffusion_transformer}"

mkdir -p "${OUTPUT_DIR}"
cd "${POLICY_DIR}"

echo "[RDT_1B] data_setting=${data_setting}"
echo "[RDT_1B] checkpoint_dir=${OUTPUT_DIR}"

RDT_DEEPSPEED_ARGS="${RDT_DEEPSPEED_ARGS:---hostfile=hostfile.txt --num_gpus=${NUM_GPUS}}"
# shellcheck disable=SC2086
deepspeed ${RDT_DEEPSPEED_ARGS} rdt/main.py \
    --deepspeed="./rdt/configs/zero2.json" \
    --pretrained_model_name_or_path="${RDT_PRETRAINED_MODEL:-robotics-diffusion-transformer/rdt-1b}" \
    --pretrained_text_encoder_name_or_path="${TEXT_ENCODER_NAME}" \
    --pretrained_vision_encoder_name_or_path="${VISION_ENCODER_NAME}" \
    --output_dir="${OUTPUT_DIR}" \
    --seed="${seed}" \
    --train_batch_size="${RDT_TRAIN_BATCH_SIZE:-32}" \
    --sample_batch_size="${RDT_SAMPLE_BATCH_SIZE:-64}" \
    --max_train_steps="${RDT_MAX_TRAIN_STEPS:-200000}" \
    --checkpointing_period="${RDT_CHECKPOINTING_PERIOD:-1000}" \
    --sample_period="${RDT_SAMPLE_PERIOD:-500}" \
    --checkpoints_total_limit="${RDT_CHECKPOINTS_TOTAL_LIMIT:-40}" \
    --lr_scheduler="${RDT_LR_SCHEDULER:-constant}" \
    --learning_rate="${RDT_LEARNING_RATE:-1e-4}" \
    --mixed_precision="${RDT_MIXED_PRECISION:-bf16}" \
    --dataloader_num_workers="${RDT_DATALOADER_NUM_WORKERS:-8}" \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr="${RDT_STATE_NOISE_SNR:-40}" \
    --load_from_hdf5 \
    --report_to="${RDT_REPORT_TO:-wandb}"
