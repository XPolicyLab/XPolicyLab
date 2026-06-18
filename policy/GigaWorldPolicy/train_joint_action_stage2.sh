#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the LeRobot v2.1 dataset directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${POLICY_DIR}/experiments}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-930}"
RUN_ID="${RUN_ID:-joint_action_stage2_$(date +%m%d_%H%M%S)}"

DATASET_NAME="${DATASET_NAME:-xpolicylab_lerobot_v21_video}"
CKPT_NAME="${CKPT_NAME:-joint_action_stage2}"
ENV_CFG_TYPE="${ENV_CFG_TYPE:-arx_x5}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-all}"
ACTION_TYPE="${ACTION_TYPE:-joint}"

VIDEOPT_CKPT="${VIDEOPT_CKPT:?Set VIDEOPT_CKPT to the video-pretrained init checkpoint (model_ema.pt)}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"

if [[ ! -f "${VIDEOPT_CKPT}" ]]; then
  echo "[ERROR] VIDEOPT_CKPT not found: ${VIDEOPT_CKPT}" >&2
  exit 1
fi

if [[ -n "${CONDA_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${POLICY_DIR}/giga_world_policy/src:${POLICY_DIR}/giga_world_policy:${POLICY_DIR}/../..:${PYTHONPATH:-}"

export GIGAWORLD_CONFIG="${GIGAWORLD_CONFIG:-configs.joint_action_stage2.config}"
export GIGAWORLD_INIT_CHECKPOINT="${GIGAWORLD_INIT_CHECKPOINT:-${VIDEOPT_CKPT}}"
export GIGAWORLD_DATA_DIR="${GIGAWORLD_DATA_DIR:-${DATA_DIR}}"
export GIGAWORLD_OUTPUT_ROOT="${GIGAWORLD_OUTPUT_ROOT:-${OUTPUT_ROOT}}"
export GIGAWORLD_CKPT_DIR="${GIGAWORLD_CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${DATASET_NAME}-${CKPT_NAME}-${ENV_CFG_TYPE}-${EXPERT_DATA_NUM}-${ACTION_TYPE}-${SEED}-${RUN_ID}}"
export GIGAWORLD_NORM_PATH="${GIGAWORLD_NORM_PATH:-${DATA_DIR}/norm_stats_delta.json}"
export GIGAWORLD_NUM_FRAMES="${GIGAWORLD_NUM_FRAMES:-28}"
export GIGAWORLD_ACTION_CHUNK="${GIGAWORLD_ACTION_CHUNK:-${GIGAWORLD_NUM_FRAMES}}"
export GIGAWORLD_MAX_EPOCHS="${GIGAWORLD_MAX_EPOCHS:-5}"
export GIGAWORLD_MAX_STEPS="${GIGAWORLD_MAX_STEPS:-0}"
export GIGAWORLD_BATCH_SIZE_PER_GPU="${GIGAWORLD_BATCH_SIZE_PER_GPU:-2}"
export GIGAWORLD_GRAD_ACCUM="${GIGAWORLD_GRAD_ACCUM:-2}"
export GIGAWORLD_CHECKPOINT_INTERVAL="${GIGAWORLD_CHECKPOINT_INTERVAL:-25000}"
export GIGAWORLD_CHECKPOINT_EPOCH_INTERVAL="${GIGAWORLD_CHECKPOINT_EPOCH_INTERVAL:-1}"
export GIGAWORLD_ACTION_LOSS_WEIGHT="${GIGAWORLD_ACTION_LOSS_WEIGHT:-1.0}"
export GIGAWORLD_VISUAL_LOSS_WEIGHT="${GIGAWORLD_VISUAL_LOSS_WEIGHT:-1.0}"
export GIGAWORLD_FREEZE_BACKBONE="${GIGAWORLD_FREEZE_BACKBONE:-0}"
export GIGAWORLD_FREEZE_ACTION="${GIGAWORLD_FREEZE_ACTION:-0}"
export GIGAWORLD_USE_GT_ACTION_FOR_VIDEO="${GIGAWORLD_USE_GT_ACTION_FOR_VIDEO:-0}"
export GIGAWORLD_WANDB_PROJECT="${GIGAWORLD_WANDB_PROJECT:-gwp-xpolicylab}"
export GIGAWORLD_WANDB_NAME="${GIGAWORLD_WANDB_NAME:-${RUN_ID}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MASTER_PORT="${MASTER_PORT:-29547}"
export TORCH_DISTRIBUTED_TIMEOUT_SEC="${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}"
export GIGAWORLD_DISTRIBUTED_TIMEOUT="${GIGAWORLD_DISTRIBUTED_TIMEOUT:-${TORCH_DISTRIBUTED_TIMEOUT_SEC}}"
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-60}"
export GIGAWORLD_ACCEL_CONFIG="${GIGAWORLD_ACCEL_CONFIG:-${POLICY_DIR}/giga_world_policy/scripts/accelerate_configs/config_deepspeed_zero2_video_pt_timeout.json}"

{
  echo "[GigaWorldPolicy] XPolicyLab joint action fine-tune from videopt stage1"
  echo "  run_id:       ${RUN_ID}"
  echo "  data_dir:     ${GIGAWORLD_DATA_DIR}"
  echo "  init_ckpt:    ${GIGAWORLD_INIT_CHECKPOINT}"
  echo "  ckpt_dir:     ${GIGAWORLD_CKPT_DIR}"
  echo "  log_file:     ${LOG_FILE}"
  echo "  gpu_ids:      ${GPU_IDS}"
  echo "  config:       ${GIGAWORLD_CONFIG}"
  echo "  epochs:       ${GIGAWORLD_MAX_EPOCHS}"
  echo "  num_frames:   ${GIGAWORLD_NUM_FRAMES}"
  echo "  losses:       action=${GIGAWORLD_ACTION_LOSS_WEIGHT}, visual=${GIGAWORLD_VISUAL_LOSS_WEIGHT}"
  echo "  freeze:       backbone=${GIGAWORLD_FREEZE_BACKBONE}, action=${GIGAWORLD_FREEZE_ACTION}"
  echo "  wandb:        ${GIGAWORLD_WANDB_PROJECT}/${GIGAWORLD_WANDB_NAME} (${WANDB_MODE})"
  echo "  TORCH_DIST_TIMEOUT: ${TORCH_DISTRIBUTED_TIMEOUT_SEC}s"
  echo "  DEEPSPEED_TIMEOUT:  ${DEEPSPEED_TIMEOUT}min"
  echo "  ACCEL_CONFIG:       ${GIGAWORLD_ACCEL_CONFIG}"
} | tee -a "${LOG_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

bash "${POLICY_DIR}/train.sh" \
  "${DATASET_NAME}" \
  "${CKPT_NAME}" \
  "${ENV_CFG_TYPE}" \
  "${EXPERT_DATA_NUM}" \
  "${ACTION_TYPE}" \
  "${SEED}" \
  "${GPU_IDS}"
