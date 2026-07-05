#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
HRDT_ROOT="${SCRIPT_DIR}/H_RDT"
DEMO_ENV_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

bench_name=${1}
run_name=${2}
env_cfg_type=${3}
total_episode_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}
pretrained_backbone_path=${8:-"${HRDT_ROOT}/checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin"}

if [[ -z "${bench_name}" || -z "${run_name}" || -z "${env_cfg_type}" || -z "${total_episode_num}" || -z "${action_type}" || -z "${seed}" || -z "${gpu_id}" ]]; then
    echo "Usage: bash train.sh <bench_name> <run_name> <env_cfg_type> <total_episode_num> <action_type> <seed> <gpu_id> [pretrained_backbone_path]"
    exit 1
fi

source_root="/vepfs-cnbje63de6fae220/hekun/datasets/RoboDojo/sim_cloud"
stats_path="${HRDT_ROOT}/datasets/xpolicylab/stats.json"

train_batch_size=32
sample_batch_size=16
num_processes=8
max_train_steps=1000000
checkpointing_period=5000
checkpoints_total_limit=40
dataloader_num_workers=4
learning_rate=1e-4
report_to="tensorboard"
deepspeed_config="configs/zero1.json"

task_arg="all"
task_name="${run_name}"
dataset_mode="multi_task"
task_count=35
max_episodes_per_task=$((total_episode_num / task_count))

processed_name="${bench_name}-${run_name}-${env_cfg_type}-${total_episode_num}-${action_type}"
run_root="${SCRIPT_DIR}/data/${processed_name}"
output_dir="${SCRIPT_DIR}/checkpoints/${processed_name}-${seed}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
free_port=$(bash "${UTILS_DIR}/get_free_port.sh")

echo "[H_RDT] raw_dataset=${bench_name}, run=${run_name}, tasks=${task_arg}, mode=${dataset_mode}, env_cfg=${env_cfg_type}"
echo "[H_RDT] total_episode_num=${total_episode_num}, max_episodes_per_task=${max_episodes_per_task}"
echo "[H_RDT] action_type=${action_type}, action_dim=${action_dim}, seed=${seed}, gpu=${gpu_id}"

cd "${SCRIPT_DIR}"

mkdir -p "${run_root}"
config_path="${run_root}/hrdt_finetune_xpolicy.yaml"
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

export PYTHONPATH="${DEMO_ENV_ROOT}:${ROOT_DIR}:${HRDT_ROOT}:${PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export XPOLICY_HRDT_SOURCE_ROOT="${source_root}"
export XPOLICY_HRDT_RAW_BENCH_NAME="${bench_name}"
export XPOLICY_HRDT_ENV_CFG_TYPE="${env_cfg_type}"
export XPOLICY_HRDT_ACTION_TYPE="${action_type}"
export XPOLICY_HRDT_MAX_EPISODES="${max_episodes_per_task}"
export XPOLICY_HRDT_STAT_PATH="${stats_path}"
export XPOLICY_HRDT_DATASET_MODE="${dataset_mode}"
export XPOLICY_HRDT_TASKS="${task_arg}"
export WANDB_PROJECT="hrdt"
export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

mkdir -p "${output_dir}"

cd "${HRDT_ROOT}"

pretrained_args=()
if [[ -n "${pretrained_backbone_path}" ]]; then
    pretrained_args+=(--pretrained_backbone_path="${pretrained_backbone_path}")
fi

accelerate launch --num_processes="${num_processes}" --main_process_port "${free_port}" main.py \
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
    --upsample_rate=1 \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="finetune" \
    --task_name="${task_name}" \
    --dataset_name="xpolicylab" \
    --seed="${seed}" \
    "${pretrained_args[@]}"