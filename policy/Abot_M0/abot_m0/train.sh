#!/usr/bin/env bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Checkpoint dir: checkpoints/<bench>-<ckpt>-<env_cfg>-<action>-<seed>

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  echo "Example: $0 RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

# LeRobot 数据路径；默认读取 RoboDojo abot cotrain 数据，可通过环境变量覆盖
DATA_ROOT="${ABOT_DATA_ROOT:-/mnt/xspark-data/xspark_shared/lerobot}"
DATASET_REPO="${ABOT_DATASET_REPO:-RoboDojo_sim_v21_video_abot}"
DATA_MIX="${ABOT_DATA_MIX:-robodojo_sim}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
IFS=',' read -r -a _gpus <<< "${gpu_id}"
NUM_GPUS="${#_gpus[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "gpu_id must contain at least one GPU id, got: ${gpu_id}" >&2
  exit 1
fi

export DATA_ROOT DATASET_REPO DATA_MIX
export RUN_ROOT_DIR="${POLICY_DIR}/checkpoints"
export RUN_ID="${ckpt_setting}"
export SEED="${seed}"
export NUM_GPUS

export MODEL_ROOT="${ABOT_MODEL_ROOT:-/mnt/xspark-data/xspark_shared/model_weights}"
export BASE_VLM="${ABOT_BASE_VLM:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct-Action}"
export PRETRAIN_CKPT="${ABOT_PRETRAIN_CKPT:-${MODEL_ROOT}/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt}"
export RELOAD_MODULES="${ABOT_RELOAD_MODULES:-qwen_vl_interface}"

# 数据已 prepare 过时设为空，避免覆盖多任务指令
export PREPARE_SCRIPT="${ABOT_PREPARE_SCRIPT:-}"

export BATCH_SIZE="${ABOT_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${ABOT_GRAD_ACC:-1}"
export NUM_WORKERS="${ABOT_NUM_WORKERS:-0}"
# RoboDojo_sim_v21_video_abot 为 AV1 编码，必须用 torchvision_av；decord 无法解码
export VIDEO_BACKEND="${ABOT_VIDEO_BACKEND:-torchvision_av}"
export MAX_TRAIN_STEPS="${ABOT_MAX_TRAIN_STEPS:-150000}"
export SAVE_INTERVAL="${ABOT_SAVE_INTERVAL:-10000}"

mkdir -p "${ckpt_dir}"

echo "[ABot-M0] data_setting=${data_setting}"
echo "[ABot-M0] ckpt_setting=${ckpt_setting}"
echo "[ABot-M0] dataset_root=${DATA_ROOT}/${DATASET_REPO}"
echo "[ABot-M0] checkpoint_dir=${ckpt_dir}"
echo "[ABot-M0] seed=${seed}"
echo "[ABot-M0] gpu_id=${gpu_id} (num_gpus=${NUM_GPUS})"
echo "[ABot-M0] per_device_batch_size=${BATCH_SIZE}, grad_acc=${GRADIENT_ACCUMULATION_STEPS}, num_workers=${NUM_WORKERS}, video_backend=${VIDEO_BACKEND}"
echo "[ABot-M0] effective_batch_size=$((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"

# region agent log
python - <<PY >/dev/null 2>&1 || true
import json, time
from pathlib import Path
log_path = Path("/personal/tianxing/RoboDojo/XPolicyLab/.cursor/debug-0684e4.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"sessionId":"0684e4","runId":"pre-fix","hypothesisId":"H1","location":"policy/Abot_M0/abot_m0/train.sh:args","message":"parsed Abot_M0 train args","data":{"argc":${#},"bench_name":"${bench_name}","ckpt_name":"${ckpt_name}","env_cfg_type":"${env_cfg_type}","action_type":"${action_type}","seed":"${seed}","gpu_id":"${gpu_id}","ckpt_setting":"${ckpt_setting}"},"timestamp":int(time.time()*1000)}) + "\n")
PY
# endregion agent log

bash "${POLICY_DIR}/examples/RoboDojo/train_files/run_RoboDojo_train.sh"
