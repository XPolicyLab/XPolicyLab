#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type> <gpu_id> <seed>
EOF
}

if [ "$#" -ne 7 ]; then
    usage >&2
    exit 1
fi

dataset_name="$1"
task_name="$2"
env_cfg_type="$3"
expert_data_num="$4"
action_type="$5"
gpu_id="$6"
seed="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
A1_DIR="${SCRIPT_DIR}/A1"
DEFAULT_DATA_DIR="$(cd "${ROOT_DIR}/.." && pwd)/models"
DEFAULT_PRETRAIN_CHECKPOINT="${DEFAULT_DATA_DIR}/a1-pretrain"

export SCRIPT_DIR ROOT_DIR A1_DIR

if [ -n "${A1_TRAIN_CONFIG:-}" ]; then
    CONFIG_FILE="${A1_TRAIN_CONFIG}"
elif [ -f "${A1_DIR}/train_config.local.yaml" ]; then
    CONFIG_FILE="${A1_DIR}/train_config.local.yaml"
else
    CONFIG_FILE="${A1_DIR}/train_config.yaml"
fi
if [ -f "${CONFIG_FILE}" ]; then
    eval "$(
        python3 - "${CONFIG_FILE}" <<'PY'
import os
import shlex
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

mapping = {
    "DATA_DIR": ("paths", "data_dir"),
    "PRETRAIN_CHECKPOINT": ("paths", "pretrain_checkpoint"),
    "HF_HOME": ("paths", "hf_home"),
    "XDG_CACHE_HOME": ("paths", "xdg_cache_home"),
    "HF_HUB_OFFLINE": ("paths", "hf_hub_offline"),
    "TRAIN_STEPS": ("training", "train_steps"),
    "GLOBAL_BATCH_SIZE": ("training", "global_batch_size"),
    "DEVICE_TRAIN_MICROBATCH_SIZE": ("training", "device_train_microbatch_size"),
    "MAX_CROPS": ("training", "max_crops"),
    "NUM_WORKERS": ("training", "num_workers"),
    "LOG_INTERVAL": ("training", "log_interval"),
    "SAVE_INTERVAL": ("checkpoint", "save_interval"),
    "SAVE_INTERVAL_UNSHARDED": ("checkpoint", "save_interval_unsharded"),
    "SAVE_NUM_CHECKPOINTS_TO_KEEP": ("checkpoint", "save_num_checkpoints_to_keep"),
    "SAVE_NUM_UNSHARDED_CHECKPOINTS_TO_KEEP": ("checkpoint", "save_num_unsharded_checkpoints_to_keep"),
    "EARLY_EXIT": ("model", "early_exit"),
    "TRAIN_EXIT_RANDOM_LAYER": ("model", "train_exit_random_layer"),
    "FT_CONNECTOR": ("model", "ft_connector"),
    "FT_VIT": ("model", "ft_vit"),
    "FT_LLM": ("model", "ft_llm"),
    "FT_EMBEDDING": ("model", "ft_embedding"),
    "CONNECTOR_LR": ("optimizer", "connector_lr"),
    "VIT_LR": ("optimizer", "vit_lr"),
    "LLM_LR": ("optimizer", "llm_lr"),
    "ACTION_HEAD_LR": ("optimizer", "action_head_lr"),
    "CONNECTOR_WEIGHT_DECAY": ("optimizer", "connector_weight_decay"),
    "VIT_WEIGHT_DECAY": ("optimizer", "vit_weight_decay"),
    "LLM_WEIGHT_DECAY": ("optimizer", "llm_weight_decay"),
    "ACTION_HEAD_WEIGHT_DECAY": ("optimizer", "action_head_weight_decay"),
    "ADAM_BETA1": ("optimizer", "beta1"),
    "ADAM_BETA2": ("optimizer", "beta2"),
    "WARMUP_STEPS": ("scheduler", "warmup_steps"),
    "FREEZE_STEPS": ("scheduler", "freeze_steps"),
    "SCHEDULER_ALPHA_F": ("scheduler", "alpha_f"),
    "WARMUP_MIN_LR": ("scheduler", "warmup_min_lr"),
    "TORCH_DISTRIBUTED_TIMEOUT": ("distributed", "torch_distributed_timeout"),
    "TORCH_NCCL_TRACE_BUFFER_SIZE": ("distributed", "torch_nccl_trace_buffer_size"),
    "TORCH_NCCL_DUMP_ON_TIMEOUT": ("distributed", "torch_nccl_dump_on_timeout"),
    "ENABLE_WANDB": ("wandb", "enable"),
    "WANDB_API_KEY": ("wandb", "api_key"),
    "WANDB_PROJECT": ("wandb", "project"),
    "WANDB_ENTITY": ("wandb", "entity"),
    "WANDB_RUN_NAME": ("wandb", "run_name"),
    "WANDB_MODE": ("wandb", "mode"),
    "WANDB_REQUIRED": ("wandb", "required"),
}

def get_nested(data, path):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def stringify(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return os.path.expandvars(str(value))

for env_name, path in mapping.items():
    if env_name in os.environ:
        value = os.environ[env_name]
    else:
        value = stringify(get_nested(cfg, path))
    os.environ[env_name] = value
    print(f"{env_name}={shlex.quote(value)}")
PY
    )"
fi

DATA_DIR="${DATA_DIR:-${DEFAULT_DATA_DIR}}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${DEFAULT_PRETRAIN_CHECKPOINT}}"
HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
DEVICE_TRAIN_MICROBATCH_SIZE="${DEVICE_TRAIN_MICROBATCH_SIZE:-1}"
MAX_CROPS="${MAX_CROPS:-6}"
NUM_WORKERS="${NUM_WORKERS:-auto}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
SAVE_INTERVAL_UNSHARDED="${SAVE_INTERVAL_UNSHARDED:-2000}"
SAVE_NUM_CHECKPOINTS_TO_KEEP="${SAVE_NUM_CHECKPOINTS_TO_KEEP:-0}"
SAVE_NUM_UNSHARDED_CHECKPOINTS_TO_KEEP="${SAVE_NUM_UNSHARDED_CHECKPOINTS_TO_KEEP:-1}"
EARLY_EXIT="${EARLY_EXIT:-false}"
TRAIN_EXIT_RANDOM_LAYER="${TRAIN_EXIT_RANDOM_LAYER:-false}"
FT_CONNECTOR="${FT_CONNECTOR:-false}"
FT_VIT="${FT_VIT:-false}"
FT_LLM="${FT_LLM:-false}"
FT_EMBEDDING="${FT_EMBEDDING:-lm_head}"
CONNECTOR_LR="${CONNECTOR_LR:-2e-4}"
VIT_LR="${VIT_LR:-6e-6}"
LLM_LR="${LLM_LR:-5e-5}"
ACTION_HEAD_LR="${ACTION_HEAD_LR:-5e-5}"
CONNECTOR_WEIGHT_DECAY="${CONNECTOR_WEIGHT_DECAY:-0.0}"
VIT_WEIGHT_DECAY="${VIT_WEIGHT_DECAY:-0.0}"
LLM_WEIGHT_DECAY="${LLM_WEIGHT_DECAY:-0.0}"
ACTION_HEAD_WEIGHT_DECAY="${ACTION_HEAD_WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
FREEZE_STEPS="${FREEZE_STEPS:-0}"
SCHEDULER_ALPHA_F="${SCHEDULER_ALPHA_F:-0.1}"
WARMUP_MIN_LR="${WARMUP_MIN_LR:-}"
TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-1800}"
TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-a1-xpolicylab}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_REQUIRED="${WANDB_REQUIRED:-${ENABLE_WANDB}}"

DATA_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${DATA_DIR}")"
PRETRAIN_CHECKPOINT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${PRETRAIN_CHECKPOINT}")"
HF_HOME="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${HF_HOME}")"
XDG_CACHE_HOME="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${XDG_CACHE_HOME}")"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export DATA_DIR PRETRAIN_CHECKPOINT HF_HOME HF_HUB_OFFLINE XDG_CACHE_HOME
export TORCH_DISTRIBUTED_TIMEOUT TORCH_NCCL_TRACE_BUFFER_SIZE TORCH_NCCL_DUMP_ON_TIMEOUT
export PYTHONPATH="${A1_DIR}:${PYTHONPATH:-}"
export WANDB_API_KEY WANDB_PROJECT WANDB_MODE WANDB_REQUIRED

mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${SCRIPT_DIR}/checkpoints"

IFS=',' read -ra GPU_ARRAY <<< "${gpu_id}"
NPROC="${#GPU_ARRAY[@]}"
if [ "${NUM_WORKERS}" = "auto" ]; then
    if [ "${NPROC}" -gt 1 ]; then
        NUM_WORKERS=0
    else
        NUM_WORKERS=2
    fi
fi

if [ ! -d "${PRETRAIN_CHECKPOINT}" ]; then
    if [ -d "${DEFAULT_PRETRAIN_CHECKPOINT}" ]; then
        echo "[WARN] PRETRAIN_CHECKPOINT does not exist: ${PRETRAIN_CHECKPOINT}" >&2
        echo "[WARN] Falling back to default pretrain checkpoint: ${DEFAULT_PRETRAIN_CHECKPOINT}" >&2
        PRETRAIN_CHECKPOINT="${DEFAULT_PRETRAIN_CHECKPOINT}"
        export PRETRAIN_CHECKPOINT
    else
        echo "[ERROR] PRETRAIN_CHECKPOINT does not exist: ${PRETRAIN_CHECKPOINT}" >&2
        exit 1
    fi
fi

echo "[INFO] GPU ID (to use): ${gpu_id}"
action_dim="$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")"
echo "[INFO] XPolicyLab action dim: ${action_dim}"

repo_id="${dataset_name}-${task_name}-${env_cfg_type}"
LEROBOT_OUTPUT_DIR="${SCRIPT_DIR}/data"
LEROBOT_DATA_PATH="${LEROBOT_OUTPUT_DIR}/${repo_id}"
echo "[INFO] Checking if LeRobot dataset exists at: ${LEROBOT_DATA_PATH}"
if [ -d "${LEROBOT_DATA_PATH}" ]; then
    echo "[INFO] LeRobot dataset '${repo_id}' already exists, skipping conversion."
else
    bash "${SCRIPT_DIR}/process_data.sh" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}" 30 "${LEROBOT_OUTPUT_DIR}"
fi

RUN_BASENAME="${task_name}-a1-${action_type}-${expert_data_num}eps-seed${seed}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
RUNNAME="${RUNNAME:-${RUN_BASENAME}-${RUN_TIMESTAMP}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUNNAME}}"
export WANDB_RUN_NAME

CKPT_DIR="${SCRIPT_DIR}/checkpoints/${RUNNAME}"
mkdir -p "${CKPT_DIR}/log"
echo "${CKPT_DIR}" > "${SCRIPT_DIR}/checkpoints/${RUN_BASENAME}.latest"

RUNTIME_DATASET_CFG="${A1_DIR}/configs/datasets/xpolicylab_runtime.yaml"
RUNTIME_EXPERIMENT_CFG="${A1_DIR}/configs/experiments/xpolicylab_runtime.yaml"

cat > "${RUNTIME_DATASET_CFG}" <<EOF
image_augmentation:
  enable: true
  enable_random_erasing: true
  enable_sharpening: true
  augmentation_prob: 0.5

lerobot:
  - path: ${LEROBOT_DATA_PATH}
    weight: 1.0
EOF

cat > "${RUNTIME_EXPERIMENT_CFG}" <<'EOF'
model_config: models/pretrain.yaml
dataset_config: datasets/xpolicylab_runtime.yaml
EOF

WANDB_ARGS=(--wandb_debug)
if [ "${ENABLE_WANDB}" = "true" ]; then
    WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
    fi
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "[ERROR] ENABLE_WANDB=true but WANDB_API_KEY is empty." >&2
        exit 1
    fi
fi

MASTER_PORT="$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")"
EXTRA_TRAIN_ARGS=()
is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}
if [ -n "${WARMUP_MIN_LR}" ]; then EXTRA_TRAIN_ARGS+=(--warmup_min_lr "${WARMUP_MIN_LR}"); fi
if is_true "${EARLY_EXIT}"; then EXTRA_TRAIN_ARGS+=(--early_exit); fi
if is_true "${TRAIN_EXIT_RANDOM_LAYER}"; then EXTRA_TRAIN_ARGS+=(--train_exit_random_layer); fi
if is_true "${FT_CONNECTOR}"; then EXTRA_TRAIN_ARGS+=(--ft_connector); fi
if is_true "${FT_VIT}"; then EXTRA_TRAIN_ARGS+=(--ft_vit); fi
if is_true "${FT_LLM}"; then EXTRA_TRAIN_ARGS+=(--ft_llm); fi
if [ -n "${FT_EMBEDDING}" ]; then EXTRA_TRAIN_ARGS+=(--ft_embedding "${FT_EMBEDDING}"); fi

cd "${A1_DIR}"
torchrun \
    --nnodes=1 \
    --node-rank=0 \
    --master-addr=127.0.0.1 \
    --nproc-per-node="${NPROC}" \
    --master-port="${MASTER_PORT}" \
    launch_scripts/train_vla.py \
    qwen2_7b \
    --checkpoint "${PRETRAIN_CHECKPOINT}" \
    --vision_backbone openai \
    --vla_config_path xpolicylab_runtime.yaml \
    "${WANDB_ARGS[@]}" \
    --train_steps "${TRAIN_STEPS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --save_interval_unsharded "${SAVE_INTERVAL_UNSHARDED}" \
    --save_num_checkpoints_to_keep "${SAVE_NUM_CHECKPOINTS_TO_KEEP}" \
    --save_num_unsharded_checkpoints_to_keep "${SAVE_NUM_UNSHARDED_CHECKPOINTS_TO_KEEP}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --device_train_microbatch_size "${DEVICE_TRAIN_MICROBATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_crops "${MAX_CROPS}" \
    --connector_learning_rate "${CONNECTOR_LR}" \
    --vit_learning_rate "${VIT_LR}" \
    --llm_learning_rate "${LLM_LR}" \
    --action_head_learning_rate "${ACTION_HEAD_LR}" \
    --connector_weight_decay "${CONNECTOR_WEIGHT_DECAY}" \
    --vit_weight_decay "${VIT_WEIGHT_DECAY}" \
    --llm_weight_decay "${LLM_WEIGHT_DECAY}" \
    --action_head_weight_decay "${ACTION_HEAD_WEIGHT_DECAY}" \
    --adam_beta1 "${ADAM_BETA1}" \
    --adam_beta2 "${ADAM_BETA2}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --freeze_steps "${FREEZE_STEPS}" \
    --scheduler_alpha_f "${SCHEDULER_ALPHA_F}" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    --log_interval "${LOG_INTERVAL}" \
    --seed="${seed}" \
    --save_folder="${CKPT_DIR}" \
    --save_overwrite \
    2>&1 | tee -a "${CKPT_DIR}/log/training_$(date +"%Y%m%d_%H%M").txt"

echo "[INFO] Training complete. Checkpoints saved to: ${CKPT_DIR}"
