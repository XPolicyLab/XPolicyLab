#!/bin/bash
set -euo pipefail

# XPolicyLab-compatible training wrapper for the aha-wam RoboDojo model.
# It intentionally launches only the task/model used by this policy:
#   task=robodojo_local_history_updated_kv_prior_only_16
#   model=ahawam
#
# Usage:
#   bash train.sh <bench_name> <task_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id> [num_gpus]
#
# The first five XPolicyLab arguments are accepted for interface consistency.
# Training data and stats are controlled by the local AHAWAM task yaml unless
# overridden through the AHA_WAM_* environment variables below.

bench_name=${1:?Usage: bash train.sh <bench_name> <task_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id> [num_gpus]}
task_name=${2:?}
env_cfg_type=${3:?}
expert_data_num=${4:?}
action_type=${5:?}
seed=${6:?}
gpu_id=${7:?}

if [[ $# -ge 8 ]]; then
    num_gpus=${8}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi

if [[ "${action_type}" != "joint" ]]; then
    echo "[ERROR] aha-wam RoboDojo training config is joint/qpos only; got action_type=${action_type}." >&2
    exit 1
fi
train_seed="${AHA_WAM_TRAIN_SEED:-${seed}}"
if (( train_seed <= 0 )); then
    echo "[aha-wam train] seed=${train_seed} is not accepted by AHAWAM; using train_seed=1." >&2
    train_seed=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/AHA_WAM"
AHA_WAM_PROJECT_ROOT="${AHA_WAM_PROJECT_ROOT:-${POLICY_DIR}/AHAWAM}"
TASK_CONFIG="${AHA_WAM_TASK_CONFIG:-robodojo_local_history_updated_kv_prior_only_16}"
DATASET_DIR="${AHA_WAM_TRAIN_DATASET_DIR:-/mnt/petrelfs/muyao/data/RoboDojo_lerobot_v21_video}"
DATASET_STATS_PATH="${AHA_WAM_TRAIN_DATASET_STATS_PATH:-${DATASET_DIR}/dataset_stats.json}"
TEXT_CACHE_DIR="${AHA_WAM_TEXT_EMBED_CACHE_DIR:-${DATASET_DIR}/text_embeds_cache}"
OUTPUT_ROOT="${AHA_WAM_OUTPUT_ROOT:-${POLICY_DIR}/checkpoints}"
INIT_CHECKPOINT="${AHA_WAM_INIT_CHECKPOINT:-}"
RESUME="${AHA_WAM_RESUME:-}"
RUN_ID="${RUN_ID:-aha-wam-${bench_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}}"

batch_size="${AHA_WAM_BATCH_SIZE:-8}"
gradient_accumulation_steps="${AHA_WAM_GRADIENT_ACCUMULATION_STEPS:-8}"
num_workers="${AHA_WAM_NUM_WORKERS:-8}"
num_epochs="${AHA_WAM_NUM_EPOCHS:-20}"
max_steps="${AHA_WAM_MAX_STEPS:-null}"
learning_rate="${AHA_WAM_LEARNING_RATE:-6e-6}"
save_every="${AHA_WAM_SAVE_EVERY:-2500}"
eval_every="${AHA_WAM_EVAL_EVERY:-500}"
log_every="${AHA_WAM_LOG_EVERY:-10}"
wandb_mode="${AHA_WAM_WANDB_MODE:-offline}"
wandb_enabled="${AHA_WAM_WANDB_ENABLED:-true}"

if [[ ! -d "${AHA_WAM_PROJECT_ROOT}" ]]; then
    echo "[ERROR] Missing AHAWAM project: ${AHA_WAM_PROJECT_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${AHA_WAM_PROJECT_ROOT}/configs/task/${TASK_CONFIG}.yaml" ]]; then
    echo "[ERROR] Missing task config: ${AHA_WAM_PROJECT_ROOT}/configs/task/${TASK_CONFIG}.yaml" >&2
    exit 1
fi
if [[ ! -d "${DATASET_DIR}/meta" ]]; then
    echo "[ERROR] Missing LeRobot dataset metadata: ${DATASET_DIR}/meta" >&2
    echo "Set AHA_WAM_TRAIN_DATASET_DIR to a prepared RoboDojo LeRobot v2.1 video dataset." >&2
    exit 1
fi
if [[ ! -f "${DATASET_STATS_PATH}" ]]; then
    echo "[ERROR] Missing dataset stats: ${DATASET_STATS_PATH}" >&2
    exit 1
fi
if [[ ! -d "${TEXT_CACHE_DIR}" || -z "$(find "${TEXT_CACHE_DIR}" -name '*.pt' -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] Missing T5 text embedding cache: ${TEXT_CACHE_DIR}" >&2
    echo "Use AHAWAM/scripts/precompute_text_embeds.py for ${TASK_CONFIG}, or set AHA_WAM_TEXT_EMBED_CACHE_DIR." >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export RUN_ID
export AHA_WAM_OUTPUT_ROOT="${OUTPUT_ROOT}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/petrelfs/caijisong/dualWAM/checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}:${AHA_WAM_PROJECT_ROOT}:${AHA_WAM_PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "[aha-wam train] bench_name=${bench_name} task_name=${task_name} env_cfg_type=${env_cfg_type}"
echo "[aha-wam train] project_root=${AHA_WAM_PROJECT_ROOT}"
echo "[aha-wam train] task=${TASK_CONFIG} model=ahawam"
echo "[aha-wam train] dataset=${DATASET_DIR}"
echo "[aha-wam train] stats=${DATASET_STATS_PATH}"
echo "[aha-wam train] text_cache=${TEXT_CACHE_DIR}"
echo "[aha-wam train] output_root=${OUTPUT_ROOT} run_id=${RUN_ID}"
echo "[aha-wam train] gpus=${gpu_id} nproc_per_node=${num_gpus} train_seed=${train_seed}"

cd "${AHA_WAM_PROJECT_ROOT}"

train_args=(
    "task=${TASK_CONFIG}"
    "model=ahawam"
    "seed=${train_seed}"
    "batch_size=${batch_size}"
    "gradient_accumulation_steps=${gradient_accumulation_steps}"
    "num_workers=${num_workers}"
    "num_epochs=${num_epochs}"
    "max_steps=${max_steps}"
    "learning_rate=${learning_rate}"
    "save_every=${save_every}"
    "eval_every=${eval_every}"
    "log_every=${log_every}"
    "wandb.enabled=${wandb_enabled}"
    "wandb.mode=${wandb_mode}"
    "wandb.name=aha-wam-robodojo"
    "data.train.dataset_dirs=[${DATASET_DIR}]"
    "data.val.dataset_dirs=[${DATASET_DIR}]"
    "data.train.pretrained_norm_stats=${DATASET_STATS_PATH}"
    "data.val.pretrained_norm_stats=${DATASET_STATS_PATH}"
    "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
    "data.val.text_embedding_cache_dir=${TEXT_CACHE_DIR}"
    "output_dir=${OUTPUT_ROOT}/${TASK_CONFIG}-${RUN_ID}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
    train_args+=("init_checkpoint=${INIT_CHECKPOINT}")
fi
if [[ -n "${RESUME}" ]]; then
    train_args+=("resume=${RESUME}")
fi

bash scripts/train_zero1.sh "${num_gpus}" "${train_args[@]}"
