#!/bin/bash
set -e
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>

Optional environment overrides:
  LEROBOT_DATA_PATH                 Default: <demo_root>/RobotDojo/RoboDojo_sim_arx-x5_v30
  DREAMZERO_PRETRAINED_MODEL_PATH   Default: <demo_root>/models/checkpoints/DreamZero-AgiBot
  WAN_CKPT_DIR                      Default: <demo_root>/models/checkpoints/checkpoints/Wan2.1-I2V-14B-480P
  TOKENIZER_DIR                     Default: <demo_root>/models/checkpoints/checkpoints/umt5-xxl
  DREAMZERO_PREFLIGHT_ONLY          If 1, validate dataset and weights then exit.
  DREAMZERO_DRY_RUN                 If 1, print resolved command and exit before torchrun.
EOF
}

if [ "$#" -ne 7 ]; then
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
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
DREAMZERO_DIR="${SCRIPT_DIR}/dreamzero"

default_lerobot_path="${ROOT_DIR}/RobotDojo/RoboDojo_sim_arx-x5_v30"
dataset_path="${LEROBOT_DATA_PATH:-${default_lerobot_path}}"
run_basename="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
output_dir="${SCRIPT_DIR}/checkpoints/${run_basename}"

if [ ! -f "${dataset_path}/meta/info.json" ]; then
    echo "[DreamZero train][ERROR] LeRobot dataset info.json not found: ${dataset_path}/meta/info.json"
    echo "[DreamZero train][ERROR] Set LEROBOT_DATA_PATH to a LeRobot v3 root or DreamZero-compatible dataset root."
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "${gpu_id}"
num_gpus=${#GPU_ARRAY[@]}
num_gpus=${DREAMZERO_NUM_GPUS:-${num_gpus}}

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo "[DreamZero train] dataset=${dataset_path}"
echo "[DreamZero train] output_dir=${output_dir}"
echo "[DreamZero train] gpu_id=${gpu_id}, num_gpus=${num_gpus}, action_dim=${action_dim}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HYDRA_FULL_ERROR=1
export WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"

default_models_dir="${ROOT_DIR}/models/checkpoints"
wan_ckpt_dir="${WAN_CKPT_DIR:-${default_models_dir}/checkpoints/Wan2.1-I2V-14B-480P}"
tokenizer_dir="${TOKENIZER_DIR:-${default_models_dir}/checkpoints/umt5-xxl}"
pretrained_model_path="${DREAMZERO_PRETRAINED_MODEL_PATH:-${default_models_dir}/DreamZero-AgiBot}"
max_steps="${DREAMZERO_MAX_STEPS:-5000}"
save_steps="${DREAMZERO_SAVE_STEPS:-2500}"
batch_size="${DREAMZERO_PER_DEVICE_BATCH_SIZE:-1}"
dataloader_workers="${DREAMZERO_DATALOADER_WORKERS:-1}"
image_width="${DREAMZERO_IMAGE_WIDTH:-320}"
image_height="${DREAMZERO_IMAGE_HEIGHT:-176}"
action_horizon="${DREAMZERO_ACTION_HORIZON:-24}"
num_frames="${DREAMZERO_NUM_FRAMES:-33}"
max_chunk_size="${DREAMZERO_MAX_CHUNK_SIZE:-4}"
report_to="${DREAMZERO_REPORT_TO:-${REPORT_TO:-tensorboard}}"

require_file() {
    local path="$1"
    local hint="$2"
    if [ ! -f "${path}" ]; then
        echo "[DreamZero train][ERROR] Required file not found: ${path}"
        echo "[DreamZero train][ERROR] ${hint}"
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local hint="$2"
    if [ ! -d "${path}" ]; then
        echo "[DreamZero train][ERROR] Required directory not found: ${path}"
        echo "[DreamZero train][ERROR] ${hint}"
        exit 1
    fi
}

if [ "${DREAMZERO_DRY_RUN:-0}" != "1" ]; then
    require_dir "${pretrained_model_path}" \
        "Set DREAMZERO_PRETRAINED_MODEL_PATH to the local DreamZero-AgiBot checkpoint directory."
    require_file "${wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth" \
        "Download Wan-AI/Wan2.1-I2V-14B-480P or set WAN_CKPT_DIR to its local directory."
    require_file "${wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        "Download Wan-AI/Wan2.1-I2V-14B-480P or set WAN_CKPT_DIR to its local directory."
    require_file "${wan_ckpt_dir}/Wan2.1_VAE.pth" \
        "Download Wan-AI/Wan2.1-I2V-14B-480P or set WAN_CKPT_DIR to its local directory."
    require_dir "${tokenizer_dir}" \
        "Download google/umt5-xxl locally or set TOKENIZER_DIR to its directory."

    if [ "${DREAMZERO_PREFLIGHT_ONLY:-0}" = "1" ]; then
        echo "[DreamZero train] Preflight passed."
        exit 0
    fi

    mkdir -p "${output_dir}" "${SCRIPT_DIR}/checkpoints"
    echo "${output_dir}" > "${SCRIPT_DIR}/checkpoints/${run_basename}.latest"
fi

cd "${DREAMZERO_DIR}"

TRAIN_CMD=(
torchrun --nproc_per_node "${num_gpus}" --standalone groot/vla/experiment/experiment.py
    report_to="${report_to}" \
    data=dreamzero/agibot_relative \
    wandb_project="${WANDB_PROJECT}" \
    train_architecture="${DREAMZERO_TRAIN_ARCHITECTURE:-lora}" \
    num_frames="${num_frames}" \
    action_horizon="${action_horizon}" \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block="${action_horizon}" \
    num_state_per_block=1 \
    seed="${seed}" \
    training_args.learning_rate="${DREAMZERO_LEARNING_RATE:-1e-5}" \
    training_args.deepspeed="${DREAMZERO_DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero2.json}" \
    ++action_head_cfg.config.lora_rank="${DREAMZERO_LORA_RANK:-16}" \
    ++action_head_cfg.config.lora_alpha="${DREAMZERO_LORA_ALPHA:-16}" \
    save_steps="${save_steps}" \
    training_args.warmup_ratio="${DREAMZERO_WARMUP_RATIO:-0.05}" \
    output_dir="${output_dir}" \
    per_device_train_batch_size="${batch_size}" \
    max_steps="${max_steps}" \
    weight_decay="${DREAMZERO_WEIGHT_DECAY:-1e-5}" \
    save_total_limit="${DREAMZERO_SAVE_TOTAL_LIMIT:-10}" \
    upload_checkpoints=false \
    bf16="${DREAMZERO_BF16:-true}" \
    tf32="${DREAMZERO_TF32:-true}" \
    eval_bf16="${DREAMZERO_EVAL_BF16:-true}" \
    dataloader_pin_memory=false \
    dataloader_num_workers="${dataloader_workers}" \
    image_resolution_width="${image_width}" \
    image_resolution_height="${image_height}" \
    save_lora_only="${DREAMZERO_SAVE_LORA_ONLY:-true}" \
    max_chunk_size="${max_chunk_size}" \
    frame_seqlen="${DREAMZERO_FRAME_SEQLEN:-880}" \
    save_strategy=steps \
    agibot_data_root="${dataset_path}" \
    dit_version="${wan_ckpt_dir}" \
    text_encoder_pretrained_path="${wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="${wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="${wan_ckpt_dir}/Wan2.1_VAE.pth" \
    tokenizer_path="${tokenizer_dir}" \
    pretrained_model_path="${pretrained_model_path}" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
)

if [ "${DREAMZERO_DRY_RUN:-0}" = "1" ]; then
    printf '[DreamZero train] Dry run command:'
    printf ' %q' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

"${TRAIN_CMD[@]}"

echo "[DreamZero train] Training finished. Checkpoints saved to ${output_dir}"
