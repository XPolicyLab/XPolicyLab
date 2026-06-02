#!/usr/bin/env bash
# MolmoAct2 LeRobot 微调入口（XPolicyLab 统一 7 参数）
#
# Usage:
#   bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
#
# 示例（RoboDojo 双臂 co-train，3500 episodes，8×80GB 推荐配置）:
#   bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7
#   bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0          # 单卡
#
# 环境变量（可选）:
#   MOLMOACT2_DATASET_ROOT   LeRobot 数据集根目录（含 meta/ data/ videos/）
#   MOLMOACT2_DATASET_REPO_ID  传给 --dataset.repo_id 的标识
#   MOLMOACT2_CHECKPOINT_PATH  起点权重，默认 allenai/MolmoAct2
#   MOLMOACT2_OUTPUT_ROOT        训练输出根目录，默认 /mnt/xspark-data/xspark_shared/MolmoACT2_checkpoints
#   MOLMOACT2_BATCH_SIZE       每卡 batch size，默认 32（8 卡 global batch=256）
#   MOLMOACT2_STEPS            训练步数，默认 100000
#   MOLMOACT2_SAVE_FREQ        保存间隔，默认 5000
#   MOLMOACT2_NUM_WORKERS      dataloader workers，默认 4
#   MOLMOACT2_ACTION_MODE      continuous / discrete / both，默认 continuous
#   MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY  1=只训 action expert，默认 0（co-train 全量微调）
#   MOLMOACT2_ENABLE_LORA_VLM  1=对 VLM 开 LoRA，默认 0
#   MOLMOACT2_CHUNK_SIZE       action horizon，默认 10
#   MOLMOACT2_WANDB_ENABLE     1 开启 wandb，默认 0
#   MOLMOACT2_LOCAL_CACHE_ROOT  本机 HF datasets 缓存根目录，默认 /tmp/molmoact2-cache-$(hostname)

set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${POLICY_DIR}/molmoact2/lerobot"
VENV_BIN="${LEROBOT_DIR}/.venv/bin"
VENV_PYTHON="${VENV_BIN}/python"
VENV_LEROBOT_TRAIN="${VENV_BIN}/lerobot-train"
VENV_ACCELERATE="${VENV_BIN}/accelerate"

data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
MOLMOACT2_OUTPUT_ROOT="${MOLMOACT2_OUTPUT_ROOT:-/mnt/xspark-data/xspark_shared/MolmoACT2_checkpoints}"
OUTPUT_DIR="${MOLMOACT2_OUTPUT_ROOT}/${ckpt_setting}"
JOB_NAME="${MOLMOACT2_JOB_NAME:-${ckpt_setting}}"

# 默认：RoboDojo 双臂 v30 co-train（3500 episodes / ~1.86M frames）
# 8×80GB：每卡 bs=15 → global batch=128；
MOLMOACT2_DATASET_ROOT="${MOLMOACT2_DATASET_ROOT:-/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v30}"
MOLMOACT2_DATASET_REPO_ID="${MOLMOACT2_DATASET_REPO_ID:-RoboDojo_sim_arx-x5_v30}"
MOLMOACT2_CHECKPOINT_PATH="${MOLMOACT2_CHECKPOINT_PATH:-allenai/MolmoAct2}"

MOLMOACT2_BATCH_SIZE="${MOLMOACT2_BATCH_SIZE:-16}"
MOLMOACT2_STEPS="${MOLMOACT2_STEPS:-100000}"
MOLMOACT2_SAVE_FREQ="${MOLMOACT2_SAVE_FREQ:-10000}"
MOLMOACT2_NUM_WORKERS="${MOLMOACT2_NUM_WORKERS:-4}"
MOLMOACT2_ACTION_MODE="${MOLMOACT2_ACTION_MODE:-continuous}"
MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY="${MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY:-0}"
MOLMOACT2_ENABLE_LORA_VLM="${MOLMOACT2_ENABLE_LORA_VLM:-0}"
MOLMOACT2_CHUNK_SIZE="${MOLMOACT2_CHUNK_SIZE:-10}"
MOLMOACT2_WANDB_ENABLE="${MOLMOACT2_WANDB_ENABLE:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

# 双臂 ARX-X5 v30：3 路相机 + 14 维 joint state/action
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

TRAIN_ACTION_EXPERT_FLAG="false"
if [[ "${MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY}" == "1" ]]; then
  if [[ "${MOLMOACT2_ACTION_MODE}" != "continuous" ]]; then
    echo "错误: train_action_expert_only 仅支持 action_mode=continuous" >&2
    exit 1
  fi
  TRAIN_ACTION_EXPERT_FLAG="true"
fi

LORA_VLM_FLAG="false"
if [[ "${MOLMOACT2_ENABLE_LORA_VLM}" == "1" ]]; then
  LORA_VLM_FLAG="true"
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
echo "train_expert_only:  ${TRAIN_ACTION_EXPERT_FLAG}"
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
  --policy.train_action_expert_only="${TRAIN_ACTION_EXPERT_FLAG}"
  --policy.enable_lora_vlm="${LORA_VLM_FLAG}"
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
