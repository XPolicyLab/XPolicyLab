#!/usr/bin/env bash
# WSA_Large Real_Piper sync-mode startup reference.
# RTC is intentionally unsupported for this path.

###############################################################################
# 1. GPU server: start WSA_Large Real_Piper serve

cd /path/to/WSA
conda activate wsa

# Simple bring-up path: load the text encoder and send plain text prompts.
# For lower memory / faster startup, precompute the exact prompt first and switch
# WSA_LARGE_LOAD_TEXT_ENCODER=false with WSA_LARGE_TEXT_EMBED_CACHE_DIR set.
#
# python tools/precompute_text_embeds.py \
#   --text-embedding-cache-dir /path/to/WSA_Large/text_embeds \
#   --model-cache-dir /path/to/WSA_Large/model_cache \
#   --override-instruction "Sort desktop objects and place them in designated locations." \
#   --context-len 128 \
#   --device cuda
#
# Optional: set DIFFSYNTH_MODEL_BASE_PATH=/path/to/WSA_Large/model_cache
# to keep Wan/T5/VAE files off the default ./checkpoints directory.

CHECKPOINT_DIR=/path/to/WSA_Large/real_piper/checkpoints/30000/pretrained_model \
STATS_KEY=real_piper \
ACTION_MODE=abs \
DEVICE=cuda \
LOAD_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=8102 \
INFER_HORIZON=24 \
DEFAULT_PROMPT="Sort desktop objects and place them in designated locations." \
RTC_ENABLED=false \
WSA_LARGE_LOAD_TEXT_ENCODER=true \
WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN=true \
WSA_LARGE_CONCAT_MULTI_CAMERA=horizontal \
WSA_LARGE_VIDEO_HEIGHT=224 \
WSA_LARGE_VIDEO_WIDTH=448 \
bash evaluation/Real_Piper_Example/01_serve_wsa_large_real_piper_sync.sh


###############################################################################
# 2. Robot: start WSA_Large Piper inference

source /home/admin1/deploy/piper_ros/devel/setup.bash
cd /home/admin1/WSA
conda activate deploy

INIT_POS="-90885.0 38280.0 -47982.0 518.0 68317.0 1278.0 -2100.0"

WS_HOST=192.168.1.10 \
WS_PORT=8102 \
TASK_PROMPT="Sort desktop objects and place them in designated locations." \
PUBLISH_RATE=15 \
ACTION_HORIZON=24 \
IMAGE_HISTORY_INTERVAL=15 \
MAX_STEPS=10000 \
INIT_JOINT_POSITION="${INIT_POS}" \
INIT_WAIT=true \
MANUAL_RESET=true \
FRONT_CAM_TOPIC=/ob_camera_02/color/image_raw \
WRIST_CAM_TOPIC=/ob_camera_01/color/image_raw \
JOINT_STATE_TOPIC=joint_states_single \
JOINT_CMD_TOPIC=js_cmd \
FIRST_INFERENCE_CHECK=false \
START_PROMPT=true \
JPEG_ROUNDTRIP=true \
GRIPPER_POSTPROCESS=true \
IMAGE_COLOR_MODE=auto \
EXPECTED_STATS_KEY=real_piper \
bash evaluation/Real_Piper_Example/02_infer_wsa_large_real_piper_sync.sh


###############################################################################
# Useful optional overrides

# If using text cache instead of loading the text encoder, replace the serve args:
# WSA_LARGE_LOAD_TEXT_ENCODER=false
# WSA_LARGE_TEXT_EMBED_CACHE_DIR=/path/to/WSA_Large/text_embeds
# WSA_LARGE_CONTEXT_LEN=128
#
# Set WS_HOST to the GPU server address.
# SEND_IMAGE_HEIGHT=480 SEND_IMAGE_WIDTH=640 bash evaluation/Real_Piper_Example/02_infer_wsa_large_real_piper_sync.sh
# IMAGE_COLOR_MODE=rgb bash evaluation/Real_Piper_Example/02_infer_wsa_large_real_piper_sync.sh
