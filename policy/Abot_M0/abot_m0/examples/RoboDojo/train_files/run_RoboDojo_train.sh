#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}}"
DATASET_REPO="${DATASET_REPO:-RoboDojo_sim_v21_video_abot}"
DATASET_DIR="${DATA_ROOT}/${DATASET_REPO}"
DATA_MIX="${DATA_MIX:-robodojo_sim}"
ROBOT_TYPE="${ROBOT_TYPE:-robotwin}"
TASK_TEXT="${TASK_TEXT-}"
PREPARE_SCRIPT="${PREPARE_SCRIPT-}"
if [[ -z "${PREPARE_SCRIPT+x}" ]]; then
  PREPARE_SCRIPT="examples/RoboDojo/prepare_RoboDojo_abot.py"
fi
PREPARE_TASK_TEXT="${PREPARE_TASK_TEXT-${TASK_TEXT-}}"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/model_weights}"
BASE_VLM="${BASE_VLM:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct-Action}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
export MODEL_ROOT

CONFIG_YAML="${CONFIG_YAML:-examples/RoboDojo/train_files/ABot_RoboDojo.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-checkpoints}"
RUN_ID="${RUN_ID:-RoboDojo_sim_abot_m0}"
SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-50}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
RELOAD_MODULES="${RELOAD_MODULES:-qwen_vl_interface}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#_VISIBLE_GPUS[@]}"
  if [[ "${NUM_GPUS}" != "${VISIBLE_GPU_COUNT}" ]]; then
    echo "Adjusting NUM_GPUS from ${NUM_GPUS} to ${VISIBLE_GPU_COUNT} to match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    NUM_GPUS="${VISIBLE_GPU_COUNT}"
  fi
  export CUDA_VISIBLE_DEVICES
fi

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    CUDA_HOME=/usr/local/cuda
  fi
fi
export CUDA_HOME
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"

export ABOT_SKIP_DEFAULT_MIXTURES=1
export ABOT_DATASETS_ROOT="${DATA_ROOT}"

if [[ "${DATA_MIX}" == "sim_stack_bowls" ]]; then
  export ABOT_SIM_STACK_BOWLS_REPO="${DATASET_REPO}"
else
  export ABOT_SINGLE_DATASET_REPO="${DATASET_REPO}"
  export ABOT_SINGLE_DATASET_MIX="${DATA_MIX}"
  export ABOT_SINGLE_DATASET_ROBOT_TYPE="${ROBOT_TYPE}"
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset not found: ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -d "${BASE_VLM}" ]]; then
  echo "BASE_VLM not found: ${BASE_VLM}" >&2
  echo "请先下载或指定 Qwen3-VL-4B-Instruct-Action。" >&2
  exit 1
fi

if [[ -z "${PRETRAIN_CKPT}" ]]; then
  PRETRAIN_CKPT="$(python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["MODEL_ROOT"]) / "ABot-M0-Pretrain"
candidates = sorted(root.rglob("*.pt")) if root.exists() else []
print(candidates[0] if candidates else "")
PY
)"
fi

if [[ -n "${PREPARE_SCRIPT}" ]]; then
  PREPARE_ARGS=(--dataset-dir "${DATASET_DIR}")
  if [[ -n "${PREPARE_TASK_TEXT}" ]]; then
    PREPARE_ARGS+=(--task "${PREPARE_TASK_TEXT}")
  fi
  python3 "${PREPARE_SCRIPT}" "${PREPARE_ARGS[@]}"
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

TRAIN_ARGS=(
  --config_file ABot/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_GPUS}"
  ABot/training/train.py
  --config_yaml "${CONFIG_YAML}"
  --framework.name ABot_M0
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}"
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}"
  --datasets.vla_data.include_state false
  --trainer.freeze_modules "${FREEZE_MODULES}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.eval_interval "${SAVE_INTERVAL}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
  --seed "${SEED}"
)

if [[ -n "${PRETRAIN_CKPT}" ]]; then
  if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAIN_CKPT}")
  if [[ -n "${RELOAD_MODULES}" ]]; then
    TRAIN_ARGS+=(--trainer.reload_modules "${RELOAD_MODULES}")
  fi
else
  echo "未找到 ABot-M0-Pretrain 的 .pt checkpoint，将只从 BASE_VLM 初始化训练。" >&2
fi

echo "Starting ABot training:"
echo "  data: ${DATASET_DIR}"
echo "  data_mix: ${DATA_MIX}"
echo "  robot_type: ${ROBOT_TYPE}"
echo "  run_id: ${RUN_ID}"
echo "  seed: ${SEED}"
echo "  cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "  num_gpus: ${NUM_GPUS}"
echo "  per_device_batch_size: ${BATCH_SIZE}"
echo "  num_workers: ${NUM_WORKERS}"
echo "  video_backend: ${VIDEO_BACKEND}"
echo "  gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  effective_batch_size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "  base_vlm: ${BASE_VLM}"
echo "  pretrain_ckpt: ${PRETRAIN_CKPT:-<none>}"
echo "  reload_modules: ${RELOAD_MODULES:-<none>}"
echo "  cuda_home: ${CUDA_HOME}"
echo "  output: ${OUTPUT_DIR}"

accelerate launch --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" "${TRAIN_ARGS[@]}"