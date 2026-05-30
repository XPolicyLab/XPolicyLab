set -xe
set -o pipefail

export TZ='Asia/Shanghai'

CFG=$1
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-12345}
export RUNNAME="test"

if [ -z "${RUNNAME+x}" ]; then  
    echo "[ERROR] RUNNAME is not set, please inject RUNNAME for experiment output directory" 
    exit 1
else  
    echo "RUNNAME is set to: $RUNNAME"  
fi

OUTPUT_DIR="./experiment"/${RUNNAME}
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
LOG_OUTPUT_DIR="experiment"/${RUNNAME}/"log"
if [ ! -d "$LOG_OUTPUT_DIR" ]; then
  mkdir -p "$LOG_OUTPUT_DIR"
fi

export WANDB_DISABLED=true
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LAUNCHER=pytorch
export DEBUG_MODE=true

torchrun \
  --nnodes=${WORLD_SIZE} \
  --node-rank=${RANK} \
  --master-addr=${MASTER_ADDR} \
  --nproc-per-node=${NPROC_PER_NODE} \
  --master-port=${MASTER_PORT} \
  go1/internvl/train/go1_train.py \
  --cfg ${CFG}
