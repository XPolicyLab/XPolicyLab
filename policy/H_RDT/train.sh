#!/bin/bash
set -e

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}
pretrained_backbone_path=${8:-""}

if [[ "${action_type}" != "joint" ]]; then
    echo "[H_RDT][ERROR] only action_type=joint is supported for training now."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
HRDT_ROOT="${SCRIPT_DIR}/H_RDT"

processed_name="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
processed_root="${SCRIPT_DIR}/data/${processed_name}"
stats_path="${processed_root}/stats.json"
config_path="${processed_root}/hrdt_finetune_xpolicy.yaml"
output_dir="${SCRIPT_DIR}/checkpoints/${processed_name}_seed${seed}"
lang_embedding_path="${HRDT_ROOT}/datasets/robotwin2/lang_embeddings/${task_name}.pt"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
free_port=$(bash "${UTILS_DIR}/get_free_port.sh")

echo "[H_RDT] dataset=${dataset_name}, task=${task_name}, env_cfg=${env_cfg_type}"
echo "[H_RDT] action_type=${action_type}, action_dim=${action_dim}, seed=${seed}, gpu=${gpu_id}"

cd "${SCRIPT_DIR}"

if [[ ! -d "${processed_root}/${task_name}/demo_clean/data" ]]; then
    bash process_data.sh \
        "${dataset_name}" \
        "${task_name}" \
        "${env_cfg_type}" \
        "${expert_data_num}" \
        "${action_type}"
fi

python - "${HRDT_ROOT}/configs/hrdt_finetune.yaml" "${config_path}" "${action_dim}" <<'PY'
import sys
import yaml

src, dst, action_dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)

cfg.setdefault("common", {})["state_dim"] = action_dim
cfg.setdefault("common", {})["action_dim"] = action_dim
cfg.setdefault("model", {}).setdefault("hrdt", {})["output_size"] = action_dim

with open(dst, "w", encoding="utf-8") as fp:
    yaml.safe_dump(cfg, fp, sort_keys=False)
PY

if [[ ! -f "${lang_embedding_path}" ]]; then
    default_t5_path="${HRDT_ROOT}/t5-v1_1-xxl"
    export T5_MODEL_PATH="${T5_MODEL_PATH:-${default_t5_path}}"
    export HRDT_CONFIG_PATH="${config_path}"
    if [[ ! -d "${T5_MODEL_PATH}" ]]; then
        echo "[H_RDT][ERROR] missing language embedding: ${lang_embedding_path}"
        echo "[H_RDT][ERROR] set T5_MODEL_PATH to a local t5-v1_1-xxl directory, or place ${task_name}.pt under ${HRDT_ROOT}/datasets/robotwin2/lang_embeddings"
        exit 1
    fi
    echo "[H_RDT] generating language embeddings with T5_MODEL_PATH=${T5_MODEL_PATH}"
    (cd "${HRDT_ROOT}" && python datasets/robotwin2/encode_lang_batch.py)
fi

if [[ ! -f "${lang_embedding_path}" ]]; then
    echo "[H_RDT][ERROR] failed to create language embedding: ${lang_embedding_path}"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export XPOLICY_HRDT_DATA_ROOT="${processed_root}"
export XPOLICY_HRDT_HDF5_FOLDER="demo_clean/data"
export XPOLICY_HRDT_MAX_EPISODES="${expert_data_num}"
export XPOLICY_HRDT_STAT_PATH="${stats_path}"
export WANDB_PROJECT="${WANDB_PROJECT:-hrdt}"
export HF_HOME="${HF_HOME:-/vepfs-cnbje63de6fae220/mobile/chengy/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "${output_dir}"

cd "${HRDT_ROOT}"

train_batch_size=${HRDT_TRAIN_BATCH_SIZE:-4}
sample_batch_size=${HRDT_SAMPLE_BATCH_SIZE:-4}
max_train_steps=${HRDT_MAX_TRAIN_STEPS:-1000000}
checkpointing_period=${HRDT_CHECKPOINTING_PERIOD:-5000}
checkpoints_total_limit=${HRDT_CHECKPOINTS_TOTAL_LIMIT:-40}
dataloader_num_workers=${HRDT_DATALOADER_NUM_WORKERS:-4}
learning_rate=${HRDT_LEARNING_RATE:-1e-4}
report_to=${HRDT_REPORT_TO:-tensorboard}
deepspeed_config=${HRDT_DEEPSPEED_CONFIG:-configs/zero1.json}

pretrained_args=()
if [[ -n "${pretrained_backbone_path}" ]]; then
    pretrained_args+=(--pretrained_backbone_path="${pretrained_backbone_path}")
fi

accelerate launch --main_process_port "${free_port}" main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed="${deepspeed_config}" \
    --config_path="${config_path}" \
    --output_dir="${output_dir}" \
    --train_batch_size="${train_batch_size}" \
    --sample_batch_size="${sample_batch_size}" \
    --max_train_steps="${max_train_steps}" \
    --checkpointing_period="${checkpointing_period}" \
    --checkpoints_total_limit="${checkpoints_total_limit}" \
    --lr_scheduler="constant_with_warmup" \
    --learning_rate="${learning_rate}" \
    --mixed_precision="bf16" \
    --dataloader_num_workers="${dataloader_num_workers}" \
    --dataset_type="finetune" \
    --report_to="${report_to}" \
    --upsample_rate=3 \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="finetune" \
    --task_name="${task_name}" \
    --dataset_name="robotwin_agilex" \
    --seed="${seed}" \
    "${pretrained_args[@]}"