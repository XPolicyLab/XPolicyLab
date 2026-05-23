#!/usr/bin/env bash
set -euo pipefail


REPO_ID=${1}
OUTPUT_DIR=${2}
JOB_NAME=${3}
SEED=${4}
GPU_ID=${5}
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

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
  --seed=${SEED} \