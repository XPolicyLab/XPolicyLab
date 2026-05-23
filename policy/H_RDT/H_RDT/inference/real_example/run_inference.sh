#!/bin/bash

# H-RDT Inference Script
# Set your paths and parameters here

# Model paths
## Note in particular that lang_embeddings,hrdt.yaml, stats.json in /H_RDT/inference/real_example/utils, are just examples, you need to change them to your configuration
export CONFIG_PATH="./utils/hrdt.yaml"   # TODO: change to the your own finetune config
export PRETRAINED_MODEL_PATH="./checkpoints/robotwin2/checkpoint-50000/pytorch_model.bin"
export LANG_EMBEDDINGS_PATH="./utils/lang_embeddings/grab_blue_cup.pt"  # TODO: change to the your own lang embeddings
export STAT_FILE_PATH="./utils/stats.json"   # TODO: change to the your own stats file

# Inference parameters
export CHUNK_SIZE=16
export PUBLISH_RATE=30
export MAX_STEPS=10000

# Output directories
export VIDEO_SAVE_DIR="./videos"
export IMAGE_SAVE_DIR="./images"
export ACTION_LOG_FILE="action_log.txt"

# Create output directories if they don't exist
if [ ! -d "$VIDEO_SAVE_DIR" ]; then
    mkdir -p "$VIDEO_SAVE_DIR"
    echo "Created video directory: $VIDEO_SAVE_DIR"
fi

if [ ! -d "$IMAGE_SAVE_DIR" ]; then
    mkdir -p "$IMAGE_SAVE_DIR"
    echo "Created image directory: $IMAGE_SAVE_DIR"
fi

# Run inference
python3 inference_hrdt.py \
    --config_path="$CONFIG_PATH" \
    --pretrained_model_path="$PRETRAINED_MODEL_PATH" \
    --lang_embeddings_path="$LANG_EMBEDDINGS_PATH" \
    --stat_file_path="$STAT_FILE_PATH" \
    --chunk_size=$CHUNK_SIZE \
    --publish_rate=$PUBLISH_RATE \
    --max_publish_step=$MAX_STEPS \
    --video_save_dir="$VIDEO_SAVE_DIR" \
    --image_save_dir="$IMAGE_SAVE_DIR" \
    --action_log_file="$ACTION_LOG_FILE" \
    --seed=42 \
    --video_fps=30 \
    --video_resolution="1280x720" \
    --record_combined_video

    # Optional flags (uncomment to use):
    # --reset_only \
    # --use_keyboard_end \
    # --disable_puppet_arm \
    # --use_actions_interpolation \
    # --use_robot_base 