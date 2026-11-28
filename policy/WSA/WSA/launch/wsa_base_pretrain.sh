#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

# export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# WANDB_TOKEN=${WANDB_TOKEN}
# CONDA_ROOT=${_CONDA_ROOT}
# CONDA_ENV=wsa

# source ${CONDA_ROOT}/etc/profile.d/conda.sh
# conda activate ${CONDA_ENV}

# wandb login ${WANDB_TOKEN}

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

# export CUDA_HOME="/usr/local/cuda-12.8"
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

POLICY="${POLICY:-WSA_Base}"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-}}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-nvidia/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-depth-anything/DA3-LARGE-1.1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
INTERNDATA_ROOT="${INTERNDATA_ROOT:-""}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-""}"
ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-""}"
AGIBOT_ROOT="${AGIBOT_ROOT:-""}"
EGODEX_LEROBOT_ROOT="${EGODEX_LEROBOT_ROOT:-""}"
WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-configs/weight_rules_wsa_base_pretrain.yaml}"
USE_DIST_LOADING="${USE_DIST_LOADING:-true}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-false}"
DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-3600}"

export LEROBOT_DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC}"

ACTION_TYPE=delta
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-norm_stats}"

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  find -L "${root}" -path "*/meta/info.json" 2>/dev/null \
    | while read -r info_path; do
        ds_dir="$(dirname "$(dirname "${info_path}")")"
        if [[ -d "${ds_dir}/data" || -d "${ds_dir}/videos" ]]; then
          echo "${ds_dir}"
        fi
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(
  {
    discover_dataset_dirs "${INTERNDATA_ROOT}"
    discover_dataset_dirs "${ROBOTWIN_ROOT}"
    discover_dataset_dirs "${ROBOCHALLENGE_ROOT}"
    discover_dataset_dirs "${AGIBOT_ROOT}"
    discover_dataset_dirs "${EGODEX_LEROBOT_ROOT}"
  } | sort -u
)

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No valid LeRobot datasets found."
  echo "Please set one or more of: INTERNDATA_ROOT ROBOTWIN_ROOT ROBOCHALLENGE_ROOT AGIBOT_ROOT EGODEX_LEROBOT_ROOT"
  exit 1
fi

if [[ -z "${POLICY_INIT_PATH}" ]]; then
  echo "Please set POLICY_INIT_PATH to the WSA Base bootstrap checkpoint."
  echo "For backward compatibility, PRETRAINED_PATH is also accepted."
  exit 1
fi

if [[ -n "${POLICY_INIT_PATH}" && ! -d "${POLICY_INIT_PATH}" ]]; then
  echo "POLICY_INIT_PATH does not exist or is not a directory: ${POLICY_INIT_PATH}"
  exit 1
fi

if [[ -n "${WEIGHT_RULES_PATH}" && ! -f "${WEIGHT_RULES_PATH}" ]]; then
  echo "WEIGHT_RULES_PATH does not exist: ${WEIGHT_RULES_PATH}"
  exit 1
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
  if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    echo "wsa_base_pretrain.sh is a multi-dataset script and does not accept DATASET_EXTERNAL_STATS_PATH."
    echo "Please set DATASET_EXTERNAL_STATS_ROOT instead."
    exit 1
  fi
  if [[ -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    echo "USE_EXTERNAL_STATS=true but DATASET_EXTERNAL_STATS_ROOT is not set."
    exit 1
  fi
fi

echo "Discovered ${#DATASET_REPO_IDS[@]} datasets"
echo "INTERNDATA_ROOT=${INTERNDATA_ROOT}"
echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
echo "ROBOCHALLENGE_ROOT=${ROBOCHALLENGE_ROOT}"
echo "AGIBOT_ROOT=${AGIBOT_ROOT}"
echo "EGODEX_LEROBOT_ROOT=${EGODEX_LEROBOT_ROOT}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH}"
echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
echo "COSMOS_TOKENIZER_PATH_OR_NAME=${COSMOS_TOKENIZER_PATH_OR_NAME}"
echo "DA3_MODEL_PATH_OR_NAME=${DA3_MODEL_PATH_OR_NAME}"
echo "DA3_ALIGNMENT_MODE=${DA3_ALIGNMENT_MODE}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_ROOT=${DATASET_EXTERNAL_STATS_ROOT}"
echo "USE_DIST_LOADING=${USE_DIST_LOADING}"
echo "DDP_TIMEOUT_SEC=${DDP_TIMEOUT_SEC}"
echo "WEIGHT_RULES_PATH=${WEIGHT_RULES_PATH}"

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating dataset robot_type registration and stats readiness..."
  for ds_dir in "${DATASET_REPO_IDS[@]}"; do
    info_path="${ds_dir}/meta/info.json"
    if [[ ! -f "${info_path}" ]]; then
      echo "Missing info.json: ${info_path}"
      exit 1
    fi

    robot_type="$(
      python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["robot_type"])' "${info_path}"
    )"

    if [[ -z "${robot_type}" || "${robot_type}" == "None" ]]; then
      echo "Dataset has empty robot_type: ${ds_dir}"
      exit 1
    fi

    python -c 'from lerobot.transforms.constants import MASK_MAPPING, FEATURE_MAPPING, IMAGE_MAPPING; import sys; rt=sys.argv[1]; missing=[name for name,m in [("MASK_MAPPING", MASK_MAPPING), ("FEATURE_MAPPING", FEATURE_MAPPING), ("IMAGE_MAPPING", IMAGE_MAPPING)] if rt not in m]; raise SystemExit(0 if not missing else "robot_type=" + rt + " missing in " + ", ".join(missing))' "${robot_type}"

    if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
      stat_path="${DATASET_EXTERNAL_STATS_ROOT}/${robot_type}/${ACTION_TYPE}/stats.json"
      if [[ ! -f "${stat_path}" ]]; then
        echo "Missing external stats for ${ds_dir}"
        echo "Expected: ${stat_path}"
        exit 1
      fi
    fi

    echo "  OK: ${ds_dir} -> robot_type=${robot_type}"
  done
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

BASE_OUTPUT_DIR="outputs/${POLICY}"
DATASET_NAME="multidata"
JOB_NAME="${POLICY}-${DATASET_NAME}-${ACTION_TYPE}-pretrain-$(date +'%Y_%m_%d_%H_%M_%S')"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-WSA_Base}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

config_path="xxx/checkpoints/last/pretrained_model/train_config.json"

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers=12
    --job_name="${JOB_NAME}"
    # --resume=true
    # --config_path=${config_path}

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path="${POLICY_INIT_PATH}"
    --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"
    --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5.0e-5
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps=300000
    --policy.scheduler_decay_lr=1.0e-5
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.enable_3d_queries=true
    --policy.num_3d_query_tokens=432  # 3 views x 12 x 12 query grid
    --policy.lambda_3d=0.01
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"
    --policy.log_da3_teacher_timing=true

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.video_backend=pyav

    --seed=42
    --batch_size=12
    --steps=300000
    --save_freq=10000
    --log_freq=25

    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT}"
    --wandb.mode=offline
)

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    ARGS+=(--dataset.external_stats_root="${DATASET_EXTERNAL_STATS_ROOT}")
fi

if [[ -n "${WEIGHT_RULES_PATH}" ]]; then
    ARGS+=(--dataset.weight_rules_path="${WEIGHT_RULES_PATH}")
fi

if [[ "${USE_DIST_LOADING}" == "true" ]]; then
    ARGS+=(--dataset.dist_loading=true)
fi

accelerate launch "${ARGS[@]}"
