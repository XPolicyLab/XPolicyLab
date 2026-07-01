#!/bin/bash
# Start official wan_va_server.py (WebsocketPolicyServer backend).
# Usage: bash launch_wan_va_server.sh [GPU_ID] [VA_PORT]
set -eo pipefail

GPU="${1:-0}"
VA_PORT="${2:-29536}"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
XPL_DIR="${ROOT_DIR}/XPolicyLab"
LINGBOT_VA_DIR="${POLICY_DIR}/lingbot_va"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/xspark-data/final_ckpt/Lingbot_VA/robodojo_sim_arx_x5_v21_new/checkpoint_step_1000/}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/xspark-data/xspark_shared/model_weights/lingbot-va-base/}"
CONFIG_NAME="${CONFIG_NAME:-robotwin30_train}"
MASTER_PORT="${MASTER_PORT:-29501}"

# set +u
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "${CONDA_ENV:-lingbot_va}"
# set -u

export PYTHONPATH="${ROOT_DIR}:${XPL_DIR}:${LINGBOT_VA_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "${POLICY_DIR}/prepare_merged_ckpt.py" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --merged-dir "${POLICY_DIR}/.merged_ckpt"

echo "[wan_va_server] config=${CONFIG_NAME} port=${VA_PORT} gpu=${GPU}"

cd "${LINGBOT_VA_DIR}"

exec env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${MASTER_PORT}" \
    RANK=0 \
    LOCAL_RANK=0 \
    WORLD_SIZE=1 \
    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port="${MASTER_PORT}" \
        wan_va/wan_va_server.py \
        --config-name "${CONFIG_NAME}" \
        --port "${VA_PORT}" \
        --save_root "${POLICY_DIR}/visualization"
