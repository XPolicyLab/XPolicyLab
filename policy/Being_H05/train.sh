#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>

Run process_data.sh first. Checkpoints are saved under:
  policy/Being_H05/checkpoints/<6-tuple>/

Required environment (set before training):
  BEINGH_MLLM_PATH      InternVL / MLLM backbone
  BEINGH_EXPERT_PATH    Qwen expert weights
  BEINGH_RESUME_PATH    Being-H05-2B (or your base checkpoint)

Optional:
  BEINGH_CONDA_ENV      Default: beingh
  NUM_GPUS / MAX_STEPS / SAVE_STEPS / LEARNING_RATE / ...
EOF
}

if [[ "$#" -ne 7 ]]; then
    usage >&2
    exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BEINGH_DIR="${SCRIPT_DIR}/Being-H"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

DATA_TAG="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
CKPT_RUN_ID="${DATA_TAG}-${seed}"
DATA_DIR="${SCRIPT_DIR}/data/${DATA_TAG}"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints/${CKPT_RUN_ID}"
DATASET_YAML="${BEINGH_DIR}/configs/posttrain/xpolicylab/${DATA_TAG}.yaml"
LOG_DIR="${OUTPUT_DIR}/log"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33m[INFO] GPU: ${gpu_id}, seed: ${seed}\033[0m"
echo -e "\033[33m[INFO] data tag (5-tuple): ${DATA_TAG}\033[0m"
echo -e "\033[33m[INFO] checkpoint dir (6-tuple): ${CKPT_RUN_ID}\033[0m"

if [[ "${action_type}" != "joint" ]]; then
    echo -e "\033[31m[ERROR] Only action_type=joint is supported for Being_H05.\033[0m" >&2
    exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    echo -e "\033[31m[ERROR] Missing processed data: ${DATA_DIR}\033[0m" >&2
    echo -e "\033[33m[ERROR] Run: bash process_data.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type}\033[0m"
    exit 1
fi

if [[ ! -f "${DATASET_YAML}" ]]; then
    python3 "${SCRIPT_DIR}/scripts/xpolicylab_dataset.py" prepare \
        --data-tag "${DATA_TAG}" \
        --data-path "${DATA_DIR}" \
        --expert-data-num "${expert_data_num}" \
        --action-type "${action_type}"
fi

PRETRAIN_MODEL="${BEINGH_MLLM_PATH:-}"
EXPERT_MODEL="${BEINGH_EXPERT_PATH:-}"
RESUME_PATH="${BEINGH_RESUME_PATH:-}"
for var_name in BEINGH_MLLM_PATH BEINGH_EXPERT_PATH BEINGH_RESUME_PATH; do
    val="${!var_name:-}"
    if [[ -z "${val}" || ! -e "${val}" ]]; then
        echo -e "\033[31m[ERROR] Set ${var_name} to an existing path (currently: '${val}').\033[0m" >&2
        exit 1
    fi
done

IFS=',' read -ra GPU_ARRAY <<< "${gpu_id}"
NUM_GPUS=${#GPU_ARRAY[@]}
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

MAX_STEPS="${MAX_STEPS:-150000}"
SAVE_STEPS="${SAVE_STEPS:-25000}"
SAVE_STEPS_START="${SAVE_STEPS_START:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
ACTION_CHUNK_LENGTH="${ACTION_CHUNK_LENGTH:-16}"
GRAD_ACCUM="${GRADIENT_ACCUMULATION_STEPS:-2}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cp "$0" "${OUTPUT_DIR}/train.sh" 2>/dev/null || true
echo "${OUTPUT_DIR}" > "${SCRIPT_DIR}/checkpoints/${CKPT_RUN_ID}.latest" 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${BEINGH_CONDA_ENV:-beingh}"

export PYTHONPATH="${BEINGH_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NO_ALBUMENTATIONS_UPDATE=1

cd "${BEINGH_DIR}"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo -e "\033[33m[INFO] dataset yaml: ${DATASET_YAML}\033[0m"
echo -e "\033[33m[INFO] output_dir: ${OUTPUT_DIR}\033[0m"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  BeingH/train/train.py \
  --seed "${seed}" \
  --mllm_path "${PRETRAIN_MODEL}" \
  --expert_path "${EXPERT_MODEL}" \
  --resume_from "${RESUME_PATH}" \
  --resume_model_only True \
  --layer_module Qwen3MoTDecoderLayer \
  --use_expert True \
  --use_flow_matching True \
  --llm_qk_norm True \
  --freeze_mllm "${FREEZE_MLLM:-False}" \
  --freeze_vit_mlp "${FREEZE_VIT_MLP:-False}" \
  --action_chunk_length "${ACTION_CHUNK_LENGTH}" \
  --max_num_tokens "${MAX_NUM_TOKENS:-8960}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE:-8960}" \
  --expected_num_tokens "${EXPECTED_NUM_TOKENS:-8960}" \
  --prefer_buffer_before "${PREFER_BUFFER_BEFORE:-4096}" \
  --max_buffer_size "${MAX_BUFFER_SIZE:-4}" \
  --attn_mode "${ATTN_MODE:-causal}" \
  --max_view_num "${MAX_VIEW_NUM:--1}" \
  --use_fixed_view "${USE_FIXED_VIEW:-False}" \
  --force_image_size "${FORCE_IMAGE_SIZE:-224}" \
  --down_sample_ratio "${DOWN_SAMPLE_RATIO:-0.5}" \
  --dataset_config_file "${DATASET_YAML}" \
  --save_merged_metadata "${SAVE_MERGED_META:-True}" \
  --conv_style being_h0 \
  --vision_select_layer -1 \
  --prompt_template long \
  --output_dir "${OUTPUT_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --max_steps "${MAX_STEPS}" \
  --save_model_only "${SAVE_MODEL_ONLY:-False}" \
  --save_steps "${SAVE_STEPS}" \
  --save_steps_start "${SAVE_STEPS_START}" \
  --logging_steps 10 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO:-0}" \
  --lr_scheduler cosine \
  --grad_checkpoint False \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --use_mpg "${USE_MPG:-True}" \
  --mpg_lambda "${MPG_LAMBDA:-0.1}" \
  --mpg_num_projections "${MPG_NUM_PROJECTIONS:-32}" \
  --mpg_refinement_iters "${MPG_REFINEMENT_ITERS:-1}" \
  --mpg_gate_temperature "${MPG_GATE_TEMPERATURE:-1.0}" \
  --mpg_use_stop_gradient "${MPG_USE_STOP_GRADIENT:-True}" \
  --use_training_time_rtc "${USE_TRAINING_TIME_RTC:-False}" \
  --simulated_delay "${SIMULATED_DELAY:-0}" \
  --rtc_delay_exp_weight "${RTC_DELAY_EXP_WEIGHT:-True}" \
  --use_inference_prefix_overwrite "${USE_INFERENCE_PREFIX_OVERWRITE:-True}" \
  2>&1 | tee -a "${LOG_FILE}"

echo -e "\033[32m[INFO] Training complete: ${OUTPUT_DIR}\033[0m"
