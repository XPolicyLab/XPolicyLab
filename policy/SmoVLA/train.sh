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
REPO_ID="${SMOVLA_REPO_ID:-${data_setting}}"
OUTPUT_DIR="${POLICY_DIR}/checkpoints/${ckpt_setting}"
JOB_NAME="${SMOVLA_JOB_NAME:-${ckpt_setting}}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "[SmoVLA] repo_id=${REPO_ID}"
echo "[SmoVLA] checkpoint_dir=${OUTPUT_DIR}"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=DaMiTian/smolvla-aloha-bimanual \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[14]},"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera3":{"type":"VISUAL","shape":[3,256,256]}}' \
  --dataset.repo_id=${REPO_ID} \
  --dataset.video_backend=${VIDEO_BACKEND} \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=10 \
  --num_workers=32 \
  --wandb.enable=false \
  --policy.adapt_to_pi_aloha=false \
  --rename_map='{"observation.images.cam_high": "observation.images.camera1","observation.images.cam_left_wrist": "observation.images.camera2","observation.images.cam_right_wrist": "observation.images.camera3"}' \
  --seed=${seed}