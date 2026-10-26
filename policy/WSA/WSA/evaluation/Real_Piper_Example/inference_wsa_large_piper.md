# WSA_Large Real_Piper startup note

This note is the WSA_Large Real_Piper deployment checklist. Steps 2-4 are
shared with the WSABase Piper note and are copied here for convenience.

RTC is intentionally unsupported for this path.

## 1. GPU server: start WSA_Large Real_Piper serve

Run this on the GPU machine. The robot-side client below should connect to this
machine's IP and port.

```bash
cd /path/to/WSA
conda activate wsa

# Simple bring-up path: load the text encoder and send plain text prompts.
# For lower memory / faster startup, precompute the exact prompt first and switch
# WSA_LARGE_LOAD_TEXT_ENCODER=false with WSA_LARGE_TEXT_EMBED_CACHE_DIR set.
#
python tools/precompute_text_embeds.py \
  --text-embedding-cache-dir /path/to/text_embeds \
  --model-cache-dir /path/to/model_cache \
  --override-instruction "Position red block, green block, and blue block from left to right in the specified sequence." \
  --context-len 128 \
  --device cuda

# Optional: set DIFFSYNTH_MODEL_BASE_PATH=/path/to/WSA_Large/model_cache
# to keep Wan/T5/VAE files off the default ./checkpoints directory.

CHECKPOINT_DIR=/path/to/wsa_large_real_piper_checkpoint \
STATS_KEY=real_piper \
ACTION_MODE=delta \
DEVICE=cuda \
LOAD_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=9103 \
INFER_HORIZON=32 \
DEFAULT_PROMPT="Position red block, green block, and blue block from left to right in the specified sequence." \
RTC_ENABLED=false \
WSA_LARGE_LOAD_TEXT_ENCODER=false \
WSA_LARGE_TEXT_EMBED_CACHE_DIR=/path/to/text_embeds \
WSA_LARGE_CONTEXT_LEN=128 \
WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN=true \
WSA_LARGE_CONCAT_MULTI_CAMERA=horizontal \
WSA_LARGE_VIDEO_HEIGHT=224 \
WSA_LARGE_VIDEO_WIDTH=448 \
bash evaluation/Real_Piper_Example/01_serve_wsa_large_real_piper_sync.sh
```

## 2. Robot: start roscore

```bash
roscore
```

## 3. Robot: start Orbbec cameras

```bash
conda activate deploy
source /home/admin1/WSA/Deploy/tmp/OrbbecSDK_develop/orbbec_ws/devel/setup.bash
roslaunch orbbec_camera multi_camera.launch
```

## 4. Robot: start Piper control node

Check CAN first:

```bash
cd /home/admin1/WSA/Deploy/Piper_ros
bash can_activate.sh
```

Then start the Piper node in the same window:

```bash
cd /home/admin1/WSA/Deploy/gui_4_2
source /home/admin1/WSA/Deploy/Piper_ros/devel/setup.bash
conda activate deploy
roslaunch piper start_single_piper.launch can_port:=can0 auto_enable:=true
```

## 5. Robot: start WSA_Large Piper inference

```bash
source /home/admin1/deploy/piper_ros/devel/setup.bash
cd /home/admin1/WSA
conda activate deploy

INIT_POS="-90885.0 38280.0 -47982.0 518.0 68317.0 1278.0 -2100.0"

WS_HOST=192.168.1.10 \
WS_PORT=9103 \
TASK_PROMPT="Position red block, green block, and blue block from left to right in the specified sequence." \
PUBLISH_RATE=24 \
ACTION_HORIZON=32 \
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
JPEG_ROUNDTRIP=false \
GRIPPER_POSTPROCESS=true \
IMAGE_COLOR_MODE=auto \
EXPECTED_STATS_KEY=real_piper \
bash evaluation/Real_Piper_Example/02_infer_wsa_large_real_piper_sync.sh
```

Set `WS_HOST` to the GPU server address. Keep `WS_PORT=9103` unless the server
command in step 1 uses a different `PORT`.

## Useful checks

```bash
rostopic list
rostopic echo /joint_states_single
rostopic echo -n 1 /ob_camera_02/color/image_raw/encoding
rostopic echo -n 1 /ob_camera_01/color/image_raw/encoding
```

Manual Piper command:

```bash
rostopic pub -1 /js_cmd sensor_msgs/JointState "{name: ['joint0','joint1','joint2','joint3','joint4','joint5','joint6'], position: [-90346,36605,-46908,831,66802,1428,100000]}"
```

## Notes

- The WSA_Large `real_piper` path should stay 7D; the server checks
  `stats_key=real_piper` and the robot client expects the matching action
  layout.
- The model-side image convention is RGB. `IMAGE_COLOR_MODE=auto` reads the ROS
  message encoding: `rgb8` is kept as RGB, `bgr8` is converted to RGB, and
  unknown encodings fall back to the legacy BGR-to-RGB path. Only force
  `IMAGE_COLOR_MODE=bgr` if the Orbbec topic encoding is wrong or missing but
  the array is known to be BGR.
- `INIT_POS` is the `rank_block_rgb` initial position.
- With `MANUAL_RESET=true` and `INIT_JOINT_POSITION` set, press Enter during
  inference to move back to `INIT_POS` and pause the current rollout. Press
  Enter again after resetting the scene to clear stale actions and start fresh
  inference from timestep 0.
- The inference window sources `/home/admin1/deploy/piper_ros/devel/setup.bash`.
  The Piper control-node window sources
  `/home/admin1/WSA/Deploy/Piper_ros/devel/setup.bash`.
