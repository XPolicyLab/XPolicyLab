### WSA_Large Real_Lift2 sync-mode startup reference
### RTC is intentionally not enabled for this path yet.

### 1. On the GPU server, start the WSA_Large Real_Lift2 serve
cd /path/to/WSA

conda activate wsa

### Optional one-time text cache for deployment without loading the T5 text encoder.
### Reuse the training TEXT_EMBED_CACHE_DIR if it already contains this task prompt.
python tools/precompute_text_embeds.py \
  --text-embedding-cache-dir /path/to/text_embeds \
  --model-cache-dir /path/to/model_cache \
  --override-instruction "Put the plastic bottle into the trash bin." \
  --context-len 128 \
  --device cuda
### Optional: set DIFFSYNTH_MODEL_BASE_PATH=/path/to/WSA_Large/model_cache
### to keep Wan/T5/VAE files off the default ./checkpoints directory.

CHECKPOINT_DIR=/path/to/wsa_large_real_lift2_checkpoint \
STATS_KEY=real_lift2 \
ACTION_MODE=delta \
DEVICE=cuda \
LOAD_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=9102 \
INFER_HORIZON=32 \
DEFAULT_PROMPT="Put the plastic bottle into the trash bin." \
RTC_ENABLED=false \
WSA_LARGE_LOAD_TEXT_ENCODER=false \
WSA_LARGE_TEXT_EMBED_CACHE_DIR=/path/to/text_embeds \
WSA_LARGE_CONTEXT_LEN=128 \
WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN=true \
bash evaluation/Real_Lift2_Example/01_serve_wsa_large_real_lift2.sh


### 2. On the robot inference machine, run the existing sync client
cd /path/to/WSA

RUN_ENV=act \
WS_URL=ws://192.168.1.10:9102 \
PROMPT="Put the plastic bottle into the trash bin." \
FRAME_RATE=24 \
IMAGE_HISTORY_INTERVAL=15 \
INFERENCE_MODE=sync \
LOG_TIMING_EVERY=5 \
bash evaluation/Real_Lift2_Example/02_inference_lift2.sh


### 3. If only the last inference window was stopped, restart just that window
cd /path/to/WSA
source ~/.bashrc
conda activate act

REAL_LIFT2_RUNTIME_ROOT=/path/to/real_lift2_runtime \
WS_URL=ws://192.168.1.10:9102 \
PROMPT="Put the plastic bottle into the trash bin." \
FRAME_RATE=24 \
IMAGE_HISTORY_INTERVAL=15 \
MAX_PUBLISH_STEP=10000 \
RECORD_MODE=Speed \
USE_BASE=true \
FIXED_BODY_HEIGHT=16 \
STATE_DIM=14 \
ACTION_DIM=14 \
INFERENCE_MODE=sync \
LOG_TIMING_EVERY=5 \
SAFE_STOP_HOME_ARMS=true \
SAFE_STOP_HOME_PUBLISH_STEPS=180 \
SAFE_STOP_BODY_HEIGHT=0 \
SAFE_STOP_PUBLISH_STEPS=30 \
bash evaluation/Real_Lift2_Example/run_real_lift2_inference.sh
