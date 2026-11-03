#!/bin/bash
set -euo pipefail

# XPolicyLab-standard training wrapper for PICA_WAM.
#
# Post-trains InternW0-Delta-Base with the PICA-WAM cross-appearance self-alignment
# (vendored task robotwin_pica) on the dataset written by process_data.sh,
# and writes the run to:
#   <POLICY_DIR>/checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/
#
# The released checkpoint used 32 GPUs x 32 samples (global batch 1024) for
# 5370 steps (10 epochs) with seed 42. Fewer GPUs keep that global batch through
# gradient accumulation.
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [num_gpus]
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK,
# MASTER_ADDR and MASTER_PORT set.
#
# Optional environment:
#   PICA_WAM_GLOBAL_BATCH (1024), PICA_WAM_MICRO_BATCH (32)
#   PICA_WAM_DATASET_DIR / PICA_WAM_TEXT_CACHE   as in process_data.sh
#   WAM_PRETRAIN_CHECKPOINT  InternW0-Delta-Base weights (default: checkpoints/pretrain.pt)
#   PICA_WAM_TRAIN_OVERRIDES extra Hydra overrides, space separated

bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [num_gpus]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_id=${6:?}
IFS=',' read -r -a gpu_arr <<< "${gpu_id}"
num_gpus=${7:-${#gpu_arr[@]}}

if [[ "${bench_name}" != "RoboTwin" || "${env_cfg_type}" != "aloha_agilex" || "${action_type}" != "joint" ]]; then
    echo "[PICA_WAM] ERROR: the recipe is RoboTwin / aloha_agilex / joint (14D), got ${bench_name} / ${env_cfg_type} / ${action_type}." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
RUNTIME_ROOT="${POLICY_DIR}/internw0_pica"
ASSET_ROOT="${INTERNW0_ASSET_ROOT:-${POLICY_DIR}/assets}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
dataset_dir="${PICA_WAM_DATASET_DIR:-${POLICY_DIR}/data/${data_setting}}"
text_cache="${PICA_WAM_TEXT_CACHE:-${POLICY_DIR}/.cache/text_embed/${data_setting}}"
wan_dir="${WAM_WAN22_PATH:-${ASSET_ROOT}/Wan-AI/Wan2.2-TI2V-5B}"
vlm_dir="${WAM_RYNNBRAIN_PATH:-${ASSET_ROOT}/Alibaba-DAMO-Academy/RynnBrain1.1-2B}"
pretrain_ckpt="${WAM_PRETRAIN_CHECKPOINT:-${POLICY_DIR}/checkpoints/pretrain.pt}"
task="robotwin_pica"

for required in "${dataset_dir}/meta/episodes.jsonl" "${text_cache}" "${wan_dir}/Wan2.2_VAE.pth" "${vlm_dir}" "${pretrain_ckpt}"; do
    if [[ ! -e "${required}" ]]; then
        echo "[PICA_WAM] ERROR: missing ${required}." >&2
        echo "[PICA_WAM] Run process_data.sh and download_assets.sh --train first." >&2
        exit 1
    fi
done

nnodes="${NNODES:-1}"
world_size=$((nnodes * num_gpus))
global_batch="${PICA_WAM_GLOBAL_BATCH:-1024}"
micro_batch="${PICA_WAM_MICRO_BATCH:-32}"
if (( global_batch % (micro_batch * world_size) != 0 )); then
    echo "[PICA_WAM] ERROR: global batch ${global_batch} is not divisible by ${micro_batch} x ${world_size} GPUs." >&2
    exit 1
fi
grad_accum=$((global_batch / (micro_batch * world_size)))

if [[ -n "${MASTER_PORT:-}" ]]; then
    master_port="${MASTER_PORT}"
elif [[ -x "${UTILS_DIR}/get_free_port.sh" ]]; then
    master_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
else
    master_port=29500
fi
mkdir -p "${ckpt_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export ROBOTWIN_DATA_ROOT="${dataset_dir}"
export ROBOTWIN_TEXT_CACHE="${text_cache}"
export WAM_PRETRAIN_CHECKPOINT="${pretrain_ckpt}"
export WAM_VLM_PATH="${vlm_dir}"
export WAM_CHECKPOINT_ROOT="${POLICY_DIR}/checkpoints"
export WAM_CACHE_ROOT="${WAM_CACHE_ROOT:-${POLICY_DIR}/.cache}"
export WAM_OUTPUT_ROOT="${POLICY_DIR}/runs"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${RUNTIME_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# DeepSpeed ZeRO-2 through Accelerate, as in the released run.
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${RUNTIME_ROOT}/scripts/ds_configs/ds_zero2_config.json"
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false
export ACCELERATE_CONFIG_DS_FIELDS=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[PICA_WAM train] task=${task} bench=${bench_name} ckpt=${ckpt_name} env=${env_cfg_type} action=${action_type} seed=${seed}"
echo "[PICA_WAM train] dataset=${dataset_dir} text_cache=${text_cache}"
echo "[PICA_WAM train] output_dir=${ckpt_dir} pretrain=${pretrain_ckpt}"
echo "[PICA_WAM train] nodes=${nnodes} gpus_per_node=${num_gpus} micro_batch=${micro_batch} grad_accum=${grad_accum} global_batch=${global_batch}"

train_overrides=(
    "task=${task}"
    "output_dir=${ckpt_dir}"
    "seed=${seed}"
    "batch_size=${micro_batch}"
    "gradient_accumulation_steps=${grad_accum}"
    "model.model_id=${wan_dir}"
    "model.tokenizer_model_id=${wan_dir}"
    "model.redirect_common_files=false"
)
if [[ -n "${PICA_WAM_TRAIN_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    train_overrides+=(${PICA_WAM_TRAIN_OVERRIDES})
fi

cd "${RUNTIME_ROOT}"
exec python -m torch.distributed.run \
    --nnodes="${nnodes}" --nproc-per-node="${num_gpus}" --node-rank="${NODE_RANK:-0}" \
    --rdzv-backend=static --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port="${master_port}" \
    --rdzv-id="${ckpt_setting}" --max-restarts=0 \
    scripts/train.py "${train_overrides[@]}"
