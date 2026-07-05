#!/bin/bash
set -e
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>

Optional environment overrides:
  LEROBOT_DATA_PATH                 Default: <demo_root>/RobotDojo/RoboDojo_sim_arx-x5_v30
  DREAMZERO_PRETRAINED_MODEL_PATH   Default: ./checkpoints/DreamZero-AgiBot, or ./checkpoints for flat layout
  WAN_CKPT_DIR                      Default: ./checkpoints/Wan2.1-I2V-14B-480P
  TOKENIZER_DIR                     Default: ./checkpoints/umt5-xxl, or Wan2.1 nested tokenizer fallback
  DREAMZERO_PREFLIGHT_ONLY          If 1, validate dataset and weights then exit.
  DREAMZERO_DRY_RUN                 If 1, print resolved command and exit before torchrun.
EOF
}

if [ "$#" -ne 7 ]; then
    usage >&2
    exit 1
fi

bench_name=$1
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
export DREAMZERO_DIR

default_lerobot_path="${ROOT_DIR}/RobotDojo/RoboDojo_sim_arx-x5_v30"
dataset_path="${LEROBOT_DATA_PATH:-${default_lerobot_path}}"
run_basename="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
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
export PYTHONPATH="${DREAMZERO_DIR}:${SCRIPT_DIR}:${ROOT_DIR}/XPolicyLab:${ROOT_DIR}:${PYTHONPATH:-}"

checkpoints_dir="${SCRIPT_DIR}/checkpoints"
default_pretrained_model_path="${checkpoints_dir}/DreamZero-AgiBot"
if [ ! -d "${default_pretrained_model_path}" ] && [ -f "${checkpoints_dir}/config.json" ]; then
    default_pretrained_model_path="${checkpoints_dir}"
fi

default_tokenizer_dir="${checkpoints_dir}/umt5-xxl"
if [ ! -d "${default_tokenizer_dir}" ] && [ -d "${checkpoints_dir}/Wan2.1-I2V-14B-480P/google/umt5-xxl" ]; then
    default_tokenizer_dir="${checkpoints_dir}/Wan2.1-I2V-14B-480P/google/umt5-xxl"
fi

wan_ckpt_dir="${WAN_CKPT_DIR:-${checkpoints_dir}/Wan2.1-I2V-14B-480P}"
tokenizer_dir="${TOKENIZER_DIR:-${default_tokenizer_dir}}"
pretrained_model_path="${DREAMZERO_PRETRAINED_MODEL_PATH:-${default_pretrained_model_path}}"
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
native_dojo_action="${DREAMZERO_NATIVE_DOJO_ACTION:-false}"
data_config="${DREAMZERO_DATA_CONFIG:-dreamzero/agibot_relative}"
if [ "${native_dojo_action}" = "1" ] || [ "${native_dojo_action}" = "true" ]; then
    native_dojo_action=true
    data_config="${DREAMZERO_DATA_CONFIG:-dreamzero/robodojo_native_relative}"
fi
python_cmd="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [ -z "${python_cmd}" ]; then
    echo "[DreamZero train][ERROR] Python executable not found. Activate the dreamzero conda env first."
    exit 1
fi

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

"${python_cmd}" - <<'PY'
import importlib.util
import os
import sys

repo_dreamzero = os.path.realpath(os.environ["DREAMZERO_DIR"])
spec = importlib.util.find_spec("groot")
origin = os.path.realpath(spec.origin) if spec and spec.origin else "<not found>"
print(f"[DreamZero train] groot package source: {origin}")
if not origin.startswith(repo_dreamzero + os.sep):
    print(
        f"[DreamZero train][ERROR] groot resolves outside this repo. "
        f"Expected under {repo_dreamzero}, got {origin}.",
        file=sys.stderr,
    )
    sys.exit(1)

expected_files = (
    "groot/vla/experiment/base.py",
    "groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py",
)
for relative_path in expected_files:
    path = os.path.join(repo_dreamzero, relative_path)
    print(f"[DreamZero train] expected source: {path}")
    if not os.path.isfile(path):
        print(f"[DreamZero train][ERROR] Required source file missing: {path}", file=sys.stderr)
        sys.exit(1)
PY

TRAIN_CMD=(
torchrun --nproc_per_node "${num_gpus}" --standalone groot/vla/experiment/experiment.py
    report_to="${report_to}" \
    data="${data_config}" \
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
    ++action_head_cfg.config.defer_lora_injection=true \
    ++action_head_cfg.config.native_dojo_action="${native_dojo_action}"
)

if [ "${DREAMZERO_DRY_RUN:-0}" = "1" ]; then
    printf '[DreamZero train] Dry run command:'
    printf ' %q' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

"${TRAIN_CMD[@]}"

echo "[DreamZero train] Training finished. Checkpoints saved to ${output_dir}"
