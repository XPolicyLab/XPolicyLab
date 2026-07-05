#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${POLICY_DIR}/weights/RDT"

export TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-${WEIGHTS_DIR}/t5-v1_1-xxl}"
export VISION_ENCODER_NAME="${VISION_ENCODER_NAME:-${WEIGHTS_DIR}/siglip-so400m-patch14-384}"
export RDT_PRETRAINED_MODEL="${RDT_PRETRAINED_MODEL:-${WEIGHTS_DIR}/rdt-1b}"
export RDT_DATASET_NAME="${RDT_DATASET_NAME:-robodojo_aloha_hdf5}"

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
OUTPUT_DIR="${POLICY_DIR}/checkpoints/${ckpt_setting}"
NUM_GPUS="$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export RDT_HDF5_DIR="${POLICY_DIR}/data/${data_setting}"
export RDT_LANG_EMBED_DIR="${POLICY_DIR}/lang_embeds"
export CFLAGS="${CFLAGS:--I/usr/include}"
export LDFLAGS="${LDFLAGS:--L/usr/lib/x86_64-linux-gnu}"
export WANDB_PROJECT="${WANDB_PROJECT:-robotics_diffusion_transformer}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RDT_PRECOMP_LANG_EMBED:-1}" == "1" ]]; then
  EMBED_DIR="${POLICY_DIR}/lang_embeds/${data_setting}"
  if [[ ! -e "${EMBED_DIR}" ]]; then
    echo "[RDT_1B] Missing lang_embeds/${data_setting}" >&2
    echo "[RDT_1B] Run first: bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}" >&2
    exit 1
  fi
fi

if [[ ! -d "${RDT_HDF5_DIR}" ]]; then
  echo "[RDT_1B] Missing data/${data_setting}" >&2
  echo "[RDT_1B] Run first: bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}" >&2
  exit 1
fi

RDT_ROOT="${POLICY_DIR}/rdt"
cd "${RDT_ROOT}"

echo "[RDT_1B] data_setting=${data_setting}"
echo "[RDT_1B] checkpoint_dir=${OUTPUT_DIR}"
echo "[RDT_1B] RDT_HDF5_DIR=${RDT_HDF5_DIR}"
echo "[RDT_1B] RDT_LANG_EMBED_DIR=${RDT_LANG_EMBED_DIR}"
echo "[RDT_1B] RDT_DATASET_NAME=${RDT_DATASET_NAME}"

RDT_DEEPSPEED_ARGS="${RDT_DEEPSPEED_ARGS:---num_gpus=${NUM_GPUS}}"
RDT_PRECOMP_LANG_EMBED_FLAG=""
if [[ "${RDT_PRECOMP_LANG_EMBED:-1}" == "1" ]]; then
  RDT_PRECOMP_LANG_EMBED_FLAG="--precomp_lang_embed"
fi
# shellcheck disable=SC2086
deepspeed ${RDT_DEEPSPEED_ARGS} main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="${RDT_PRETRAINED_MODEL}" \
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
    ${RDT_PRECOMP_LANG_EMBED_FLAG} \
    --report_to="${RDT_REPORT_TO:-wandb}"
