#!/bin/bash
set -euo pipefail

# One-click Mem_0 Execution Module training.
#
# Renders a ready-to-run config from the committed template (no manual YAML
# edits), then launches distributed training with torchrun. nproc_per_node is
# the number of GPUs you pass. Run inside the Mem_0 policy conda env.
#
# Usage:
#   bash train.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> \
#                 <action_type> <seed> <gpu_ids>
# Example (8-GPU; consumes the dataset produced by process_data.sh with the same args):
#   bash train.sh RoboDojo cover_blocks arx_x5 50 joint 42 0,1,2,3,4,5,6,7
# Smoke test on 1 GPU:
#   IS_DEBUG=true ENABLE_WANDB=false bash train.sh RoboDojo test_data arx_x5 3 joint 42 0
#
# Tunables (env): BATCH_SIZE (per-GPU, default 56), TRAIN_STEPS (default 30000),
#   ENABLE_WANDB (default true), IS_DEBUG (default false), NORM_STATS_PATH,
#   MASTER_PORT, REPO_ID (override the derived lerobot dataset path),
#   ALLOW_NO_QWEN=true (skip the Qwen backbone presence check; smoke only).

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_ids=${7}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${POLICY_DIR}/Mem_0"
ADAPTER_DIR="${UPSTREAM_DIR}/xpolicylab_adapter"

dataset_id="${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
run_name="${dataset_id}-seed${seed}"
# Paths below are relative to the upstream root (train_low.py resolves
# checkpoint_dir against PROJECT_ROOT = the upstream root).
rel_ckpt_dir="checkpoints/${run_name}"
repo_id="${REPO_ID:-${UPSTREAM_DIR}/lerobot_datasets/${dataset_id}}"
gen_config="${UPSTREAM_DIR}/${rel_ckpt_dir}/train_config.yaml"

batch_size=${BATCH_SIZE:-56}
train_steps=${TRAIN_STEPS:-30000}
enable_wandb=${ENABLE_WANDB:-true}
is_debug=${IS_DEBUG:-false}
master_port=${MASTER_PORT:-29500}

# ---- Preflight checks (fail loudly) ----
if [[ ! -d "${repo_id}" ]]; then
  echo -e "\033[31m[train] LeRobot dataset not found: ${repo_id}\033[0m" >&2
  echo "Run: bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} <M1|Mn>   (or set REPO_ID=...)" >&2
  exit 1
fi
qwen_dir="${UPSTREAM_DIR}/checkpoints/Qwen3-VL-2B-Instruct"
if [[ ! -d "${qwen_dir}" && "${ALLOW_NO_QWEN:-false}" != "true" ]]; then
  echo -e "\033[31m[train] Qwen3-VL-2B backbone not found: ${qwen_dir}\033[0m" >&2
  echo "Download it (cd Mem_0/checkpoints && python _download.py), or set ALLOW_NO_QWEN=true for a smoke run." >&2
  exit 1
fi

# Number of processes = number of GPUs in the list.
IFS=',' read -ra _gpu_arr <<< "${gpu_ids}"
nproc=${#_gpu_arr[@]}

norm_args=()
[[ -n "${NORM_STATS_PATH:-}" ]] && norm_args+=( --norm_stats_path "${NORM_STATS_PATH}" )

# ---- Render the training config ----
python "${ADAPTER_DIR}/gen_train_config.py" \
    --repo_id "${repo_id}" \
    --checkpoint_dir "${rel_ckpt_dir}" \
    --wandb_run_name "${run_name}" \
    --out "${gen_config}" \
    --seed "${seed}" \
    --batch_size "${batch_size}" \
    --train_steps "${train_steps}" \
    --enable_wandb "${enable_wandb}" \
    --is_debug "${is_debug}" \
    "${norm_args[@]}"

# ---- Launch training (torchrun from the upstream root so `source.*` imports resolve) ----
echo -e "\033[33m[train] GPUs=${gpu_ids} (nproc_per_node=${nproc}); run=${run_name}\033[0m"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
cd "${UPSTREAM_DIR}"

torchrun \
    --standalone \
    --master_port="${master_port}" \
    --nnodes=1 \
    --nproc_per_node="${nproc}" \
    source/training/train_low.py \
    --config "${gen_config}"
