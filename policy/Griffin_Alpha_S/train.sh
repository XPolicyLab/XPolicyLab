#!/usr/bin/env bash
# Fine-tune Griffin Alpha-S with stock `lerobot-train`.
#
# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra lerobot-train flags...]
#
# Starts from the fine-tune base written by process_data.sh
# (bases/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/) and trains on the LeRobot dataset of
# the same name. Checkpoints land in checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
# in lerobot-train's layout (checkpoints/<step>/pretrained_model/, plus a `last` link), which is
# what model.py loads at eval time.
#
# Environment overrides:
#   GRIFFIN_POLICY_PATH       start from another checkpoint (a directory or a Hub id such as
#                             griffinlabs/Griffin-Alpha-S-LIBERO) instead of the process_data.sh base
#   GRIFFIN_DATASET_REPO_ID   LeRobot dataset repo id (default <bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>)
#   GRIFFIN_BATCH_SIZE        default 16          GRIFFIN_STEPS       default 20000
#   GRIFFIN_SAVE_FREQ         default 5000        GRIFFIN_NUM_WORKERS default 8
#   VIDEO_BACKEND             lerobot video backend (default pyav; torchcodec needs a linkable FFmpeg)
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra lerobot-train flags...]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${GRIFFIN_CONDA_ENV:-griffin_alpha_s}"

if ! command -v lerobot-train >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
policy_path="${GRIFFIN_POLICY_PATH:-${POLICY_DIR}/bases/${data_setting}}"
dataset_repo_id="${GRIFFIN_DATASET_REPO_ID:-${data_setting}}"
output_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

if [[ -z "${GRIFFIN_POLICY_PATH:-}" && ! -f "${policy_path}/config.json" ]]; then
  echo "[Griffin_Alpha_S] no fine-tune base at ${policy_path}." >&2
  echo "[Griffin_Alpha_S] Run: bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}" >&2
  echo "[Griffin_Alpha_S] or set GRIFFIN_POLICY_PATH to a checkpoint whose processors already match the dataset." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"

echo "[Griffin_Alpha_S] policy_path=${policy_path}"
echo "[Griffin_Alpha_S] dataset_repo_id=${dataset_repo_id} (HF_LEROBOT_HOME=${HF_LEROBOT_HOME})"
echo "[Griffin_Alpha_S] output_dir=${output_dir}"

lerobot-train \
  --policy.path="${policy_path}" \
  --dataset.repo_id="${dataset_repo_id}" \
  --dataset.video_backend="${VIDEO_BACKEND:-pyav}" \
  --output_dir="${output_dir}" \
  --job_name="${ckpt_setting}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size="${GRIFFIN_BATCH_SIZE:-16}" \
  --steps="${GRIFFIN_STEPS:-20000}" \
  --save_freq="${GRIFFIN_SAVE_FREQ:-5000}" \
  --num_workers="${GRIFFIN_NUM_WORKERS:-8}" \
  --wandb.enable=false \
  --seed="${seed}" \
  "$@"
