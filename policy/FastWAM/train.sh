#!/bin/bash
set -euo pipefail

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}
if [[ $# -ge 8 ]]; then
    num_gpus=${8}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi
train_seed=${seed}
if [[ "${train_seed}" -le 0 ]]; then
    train_seed=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/FastWAM"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_DIR}/checkpoints"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
# Default dataset_id mirrors process_data.sh: per-task data_key for a single
# task, "cotrain_dataset" otherwise. Set FASTWAM_DATASET_ID to point at a
# pre-merged cotrain dataset (e.g. produced by process_data_batch.sh) without
# changing the train.sh argument shape.
data_key="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
dataset_id="${FASTWAM_DATASET_ID:-${data_key}}"
converted_root="${POLICY_DIR}/data/${dataset_id}"
dataset_dir="${converted_root}/lerobot"
stats_path="${converted_root}/dataset_stats.json"
text_cache_dir="${FASTWAM_DIR}/data/text_embeds_cache/xpolicylab/${dataset_id}"
ckpt_setting="${FASTWAM_CKPT_SETTING:-${dataset_id}-${seed}}"
action_dit="${FASTWAM_DIR}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
batch_size="${FASTWAM_BATCH_SIZE:-8}"
gradient_accumulation_steps="${FASTWAM_GRADIENT_ACCUMULATION_STEPS:-1}"
num_workers="${FASTWAM_NUM_WORKERS:-8}"
# Upstream trainer.py:39 does `int(cfg.num_epochs)` unconditionally, so even when
# the task yaml sets `num_epochs: null` (i.e. you want pure max_steps-based
# training, which `_estimate_total_train_steps()` honors), num_epochs must still
# be a non-null integer or trainer init crashes. Inject a large dummy by default;
# user can override with FASTWAM_NUM_EPOCHS.
num_epochs_override="${FASTWAM_NUM_EPOCHS:-200000}"

if [[ ! -d "${dataset_dir}/meta" ]]; then
    if [[ -n "${FASTWAM_DATASET_ID:-}" ]]; then
        echo "[ERROR] FASTWAM_DATASET_ID=${FASTWAM_DATASET_ID} but ${dataset_dir}/meta is missing."
        echo "Run process_data_batch.sh (or process_data.sh with --dataset-id) in the policy env first."
        exit 1
    fi
    bash "${POLICY_DIR}/process_data.sh" "${dataset_name}" "${task_name}" "${env_cfg_type}" "${expert_data_num}" "${action_type}"
fi

if [[ ! -f "${action_dit}" ]]; then
    echo "[ERROR] Missing ActionDiT backbone: ${action_dit}"
    echo "Run in the FastWAM policy environment:"
    echo "  cd ${FASTWAM_DIR}"
    echo "  python scripts/preprocess_action_dit_backbone.py --model-config configs/model/fastwam.yaml --output ${action_dit} --device cuda --dtype bfloat16"
    exit 1
fi

if [[ ! -d "${text_cache_dir}" || -z "$(find "${text_cache_dir}" -name '*.pt' -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] Missing real T5 text embedding cache: ${text_cache_dir}"
    echo "Run process_data.sh again in the FastWAM policy environment; it converts data and precomputes the matching text embedding cache."
    echo "  bash ${POLICY_DIR}/process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}"
    exit 1
fi

cd "${FASTWAM_DIR}"

train_common=(
    "task=robotwin_uncond_3cam_384_1e-4"
    "seed=${train_seed}"
    "batch_size=${batch_size}"
    "gradient_accumulation_steps=${gradient_accumulation_steps}"
    "num_workers=${num_workers}"
    "num_epochs=${num_epochs_override}"
    "data.train.dataset_dirs=[${dataset_dir}]"
    "data.val.dataset_dirs=[${dataset_dir}]"
    "data.train.text_embedding_cache_dir=${text_cache_dir}"
    "data.val.text_embedding_cache_dir=${text_cache_dir}"
    "data.train.pretrained_norm_stats=${stats_path}"
    "data.val.pretrained_norm_stats=${stats_path}"
    "data.train.shape_meta.action.0.raw_shape=${action_dim}"
    "data.train.shape_meta.action.0.shape=${action_dim}"
    "data.train.shape_meta.state.0.raw_shape=${action_dim}"
    "data.train.shape_meta.state.0.shape=${action_dim}"
    "data.val.shape_meta.action.0.raw_shape=${action_dim}"
    "data.val.shape_meta.action.0.shape=${action_dim}"
    "data.val.shape_meta.state.0.raw_shape=${action_dim}"
    "data.val.shape_meta.state.0.shape=${action_dim}"
    "data.train.processor.action_output_dim=${action_dim}"
    "data.train.processor.proprio_output_dim=${action_dim}"
    "data.val.processor.action_output_dim=${action_dim}"
    "data.val.processor.proprio_output_dim=${action_dim}"
    "output_dir=${POLICY_DIR}/checkpoints/${ckpt_setting}"
)

bash scripts/train_zero1.sh "${num_gpus}" "${train_common[@]}"
