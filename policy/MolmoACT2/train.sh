#!/usr/bin/env bash
# MolmoAct2 LeRobot fine-tuning entrypoint using the unified XPolicyLab 6-argument interface
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
#
# Example: RoboDojo dual-arm co-training, recommended for 8x80GB GPUs:
#   bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
#   bash train.sh RoboDojo cotrain arx_x5 joint 0 0          # single GPU
#
# Optional environment variables:
#   MOLMOACT2_DATASET_ROOT   LeRobot dataset root, including meta/, data/, and videos/
#   MOLMOACT2_DATASET_REPO_ID  Identifier passed to --dataset.repo_id
#   MOLMOACT2_CHECKPOINT_PATH  Starting checkpoint, defaults to allenai/MolmoAct2
#   MOLMOACT2_OUTPUT_ROOT        Training output root, defaults to policy/MolmoACT2/checkpoints
#   MOLMOACT2_BATCH_SIZE       Per-GPU batch size, defaults to 16 (8 GPUs gives global batch=128)
#   MOLMOACT2_STEPS            Training steps, defaults to 100000
#   MOLMOACT2_SAVE_FREQ        Save interval, defaults to 10000
#   MOLMOACT2_NUM_WORKERS      Dataloader workers, defaults to 4
#   MOLMOACT2_ACTION_MODE      continuous / discrete / both; defaults to continuous
#   MOLMOACT2_TRAIN_MODE_VLM   fft / lora / freeze for the VLM side; overrides the two legacy vars below when set
#   MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY  1 freezes the VLM (train_mode_vlm=freeze); defaults to 0 (the action expert is always fully fine-tuned)
#   MOLMOACT2_ENABLE_LORA_VLM  1 enables LoRA for the VLM (train_mode_vlm=lora); defaults to 0 for full co-training fine-tuning (fft)
#   MOLMOACT2_CHUNK_SIZE       Action horizon, defaults to 10
#   MOLMOACT2_WANDB_ENABLE     1 enables wandb; defaults to 0
#   MOLMOACT2_LOCAL_CACHE_ROOT  Local HF datasets cache root; defaults to /tmp/molmoact2-cache-$(hostname)

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${POLICY_DIR}/molmoact2/lerobot"
VENV_BIN="${LEROBOT_DIR}/.venv/bin"
VENV_PYTHON="${VENV_BIN}/python"
VENV_LEROBOT_TRAIN="${VENV_BIN}/lerobot-train"
VENV_ACCELERATE="${VENV_BIN}/accelerate"

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
MOLMOACT2_OUTPUT_ROOT="${MOLMOACT2_OUTPUT_ROOT:-${POLICY_DIR}/checkpoints}"
OUTPUT_DIR="${MOLMOACT2_OUTPUT_ROOT}/${ckpt_setting}"
JOB_NAME="${MOLMOACT2_JOB_NAME:-${ckpt_setting}}"

# Standard checkpoint path consumed by eval (model.py resolves checkpoints/<ckpt_name>).
STANDARD_CKPT_DIR="${POLICY_DIR}/checkpoints/${ckpt_setting}"
if [[ "${OUTPUT_DIR}" != "${STANDARD_CKPT_DIR}" ]]; then
  if [[ -e "${STANDARD_CKPT_DIR}" && ! -L "${STANDARD_CKPT_DIR}" ]]; then
    echo "错误: 标准 checkpoint 路径已存在且不是软链: ${STANDARD_CKPT_DIR}" >&2
    exit 1
  fi
  mkdir -p "${POLICY_DIR}/checkpoints" "${OUTPUT_DIR}"
  ln -sfn "${OUTPUT_DIR}" "${STANDARD_CKPT_DIR}"
  echo "已软链标准路径: ${STANDARD_CKPT_DIR} -> ${OUTPUT_DIR}"
fi

# Default: RoboDojo dual-arm v30 co-training
# 8x80GB: bs=15 per GPU gives global batch=128;
MOLMOACT2_DATASET_ROOT="${MOLMOACT2_DATASET_ROOT:?set MOLMOACT2_DATASET_ROOT to your RoboDojo LeRobot dataset dir}"
MOLMOACT2_DATASET_REPO_ID="${MOLMOACT2_DATASET_REPO_ID:-RoboDojo_sim_arx-x5_v30}"
MOLMOACT2_CHECKPOINT_PATH="${MOLMOACT2_CHECKPOINT_PATH:-allenai/MolmoAct2}"

MOLMOACT2_BATCH_SIZE="${MOLMOACT2_BATCH_SIZE:-16}"
MOLMOACT2_STEPS="${MOLMOACT2_STEPS:-100000}"
MOLMOACT2_SAVE_FREQ="${MOLMOACT2_SAVE_FREQ:-10000}"
MOLMOACT2_NUM_WORKERS="${MOLMOACT2_NUM_WORKERS:-4}"
MOLMOACT2_ACTION_MODE="${MOLMOACT2_ACTION_MODE:-continuous}"
MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY="${MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY:-0}"
MOLMOACT2_ENABLE_LORA_VLM="${MOLMOACT2_ENABLE_LORA_VLM:-0}"
# Optional direct override of the VLM training mode: fft / lora / freeze.
MOLMOACT2_TRAIN_MODE_VLM="${MOLMOACT2_TRAIN_MODE_VLM:-}"
MOLMOACT2_CHUNK_SIZE="${MOLMOACT2_CHUNK_SIZE:-10}"
MOLMOACT2_WANDB_ENABLE="${MOLMOACT2_WANDB_ENABLE:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

# Dual-arm ARX-X5 v30: 3 cameras plus 14-D joint state/action
IMAGE_KEYS='["observation.images.cam_high","observation.images.cam_left_wrist","observation.images.cam_right_wrist"]'
SETUP_TYPE="${MOLMOACT2_SETUP_TYPE:-dual arx x5 robotic arms in robodojo simulation}"

if [[ "${action_type}" == "joint" ]]; then
  CONTROL_MODE="${MOLMOACT2_CONTROL_MODE:-absolute joint pose}"
elif [[ "${action_type}" == "ee" ]]; then
  CONTROL_MODE="${MOLMOACT2_CONTROL_MODE:-delta end-effector pose}"
else
  CONTROL_MODE="${MOLMOACT2_CONTROL_MODE:-absolute joint pose}"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "错误: 未找到 LeRobot 训练环境 ${LEROBOT_DIR}/.venv" >&2
  echo "请先按 INSTALLATION.md 第 3 步安装:" >&2
  echo "  cd ${LEROBOT_DIR} && UV_LINK_MODE=copy uv pip install -e \".[molmoact2,training,scipy-dep]\" --index-strategy unsafe-best-match" >&2
  exit 1
fi

if [[ ! -x "${VENV_LEROBOT_TRAIN}" ]]; then
  echo "错误: 未找到 lerobot-train，请安装 training extra:" >&2
  echo "  cd ${LEROBOT_DIR} && UV_LINK_MODE=copy uv pip install -e \".[molmoact2,training,scipy-dep]\" --index-strategy unsafe-best-match" >&2
  exit 1
fi

if [[ ! -f "${MOLMOACT2_DATASET_ROOT}/meta/info.json" ]]; then
  echo "错误: LeRobot 数据集不存在: ${MOLMOACT2_DATASET_ROOT}/meta/info.json" >&2
  exit 1
fi

CODEBASE_VER="$("${VENV_PYTHON}" -c "import json; print(json.load(open('${MOLMOACT2_DATASET_ROOT}/meta/info.json'))['codebase_version'])")"
if [[ "${CODEBASE_VER}" != "v3.0" ]]; then
  echo "警告: 数据集格式为 ${CODEBASE_VER}，MolmoAct2 需要 LeRobot v3.0" >&2
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"

# LeRobot loads parquet via HuggingFace datasets, which builds pyarrow mmap cache
# under HF_DATASETS_CACHE. Keep dataset on shared storage, but use per-host local
# cache to avoid NFS lock contention when multiple nodes train concurrently.
LOCAL_CACHE_ROOT="${MOLMOACT2_LOCAL_CACHE_ROOT:-/tmp/molmoact2-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/tmp"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export TMPDIR="${TMPDIR:-${LOCAL_CACHE_ROOT}/tmp}"

IFS=',' read -ra GPU_ARR <<< "${gpu_id}"
NUM_GPUS="${#GPU_ARR[@]}"

# The action expert is always fully fine-tuned; MOLMOACT2_TRAIN_MODE_VLM (fft/lora/freeze)
# controls only the VLM side. Derive it from the legacy env vars when not set directly.
if [[ -n "${MOLMOACT2_TRAIN_MODE_VLM}" ]]; then
  TRAIN_MODE_VLM="${MOLMOACT2_TRAIN_MODE_VLM}"
elif [[ "${MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY}" == "1" ]]; then
  TRAIN_MODE_VLM="freeze"
elif [[ "${MOLMOACT2_ENABLE_LORA_VLM}" == "1" ]]; then
  TRAIN_MODE_VLM="lora"
else
  TRAIN_MODE_VLM="fft"
fi

case "${TRAIN_MODE_VLM}" in
  fft|lora|freeze) ;;
  *)
    echo "错误: MOLMOACT2_TRAIN_MODE_VLM 需为 fft / lora / freeze，当前: ${TRAIN_MODE_VLM}" >&2
    exit 1
    ;;
esac

if [[ "${TRAIN_MODE_VLM}" == "freeze" && "${MOLMOACT2_ACTION_MODE}" != "continuous" ]]; then
  echo "错误: train_mode_vlm=freeze 仅支持 action_mode=continuous" >&2
  exit 1
fi

WANDB_FLAG="false"
if [[ "${MOLMOACT2_WANDB_ENABLE}" == "1" ]]; then
  WANDB_FLAG="true"
fi

GLOBAL_BATCH_SIZE=$((MOLMOACT2_BATCH_SIZE * NUM_GPUS))

echo "=== MolmoAct2 训练 ==="
echo "data_setting:       ${data_setting}"
echo "checkpoint_dir:     ${OUTPUT_DIR}"
echo "local_cache_root:   ${LOCAL_CACHE_ROOT}"
echo "dataset.root:       ${MOLMOACT2_DATASET_ROOT}"
echo "dataset.repo_id:    ${MOLMOACT2_DATASET_REPO_ID}"
echo "base_checkpoint:    ${MOLMOACT2_CHECKPOINT_PATH}"
echo "gpus:               ${gpu_id} (${NUM_GPUS} proc)"
echo "batch_size/gpu:     ${MOLMOACT2_BATCH_SIZE}"
echo "global_batch_size:  ${GLOBAL_BATCH_SIZE}"
echo "action_mode:        ${MOLMOACT2_ACTION_MODE}"
echo "train_mode_vlm:     ${TRAIN_MODE_VLM}"
echo "chunk_size:         ${MOLMOACT2_CHUNK_SIZE}"
echo "steps:              ${MOLMOACT2_STEPS}"

cd "${LEROBOT_DIR}"
export PATH="${VENV_BIN}:${PATH}"

COMMON_ARGS=(
  --dataset.repo_id="${MOLMOACT2_DATASET_REPO_ID}"
  --dataset.root="${MOLMOACT2_DATASET_ROOT}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --dataset.image_transforms.enable=true
  --policy.type=molmoact2
  --policy.checkpoint_path="${MOLMOACT2_CHECKPOINT_PATH}"
  --policy.device=cuda
  --policy.action_mode="${MOLMOACT2_ACTION_MODE}"
  --policy.chunk_size="${MOLMOACT2_CHUNK_SIZE}"
  --policy.n_action_steps="${MOLMOACT2_CHUNK_SIZE}"
  --policy.setup_type="${SETUP_TYPE}"
  --policy.control_mode="${CONTROL_MODE}"
  --policy.image_keys="${IMAGE_KEYS}"
  --policy.model_dtype=bfloat16
  --policy.num_flow_timesteps=8
  --policy.gradient_checkpointing=true
  --policy.freeze_embedding=true
  --policy.normalize_gripper=false
  --policy.enable_knowledge_insulation=false
  --policy.train_mode_vlm="${TRAIN_MODE_VLM}"
  --policy.push_to_hub=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --steps="${MOLMOACT2_STEPS}"
  --batch_size="${MOLMOACT2_BATCH_SIZE}"
  --num_workers="${MOLMOACT2_NUM_WORKERS}"
  --log_freq=20
  --eval_freq=-1
  --save_checkpoint=true
  --save_freq="${MOLMOACT2_SAVE_FREQ}"
  --seed="${seed}"
  --wandb.enable="${WANDB_FLAG}"
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  "${VENV_ACCELERATE}" launch \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    "${COMMON_ARGS[@]}"
else
  "${VENV_LEROBOT_TRAIN}" "${COMMON_ARGS[@]}"
fi

echo ""
echo "=== 训练完成 ==="
echo "Checkpoint: ${OUTPUT_DIR}"
