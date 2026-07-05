#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

# Batch config (single source of truth):
#   global_batch = DM0_BATCH_SIZE * NUM_GPUS * DM0_GRAD_ACCUM
export DM0_GLOBAL_BATCH_SIZE="${DM0_GLOBAL_BATCH_SIZE:-64}"
export DM0_BATCH_SIZE="${DM0_BATCH_SIZE:-4}"
export DM0_MAX_STEPS="${DM0_MAX_STEPS:-100000}"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEXBOTIC_ROOT="${POLICY_DIR}/dexbotic"
EXP_SCRIPT="${DEXBOTIC_ROOT}/playground/benchmarks/robodojo/robodojo_dm0.py"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
converted_data_root="${DM0_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
bench_name_registered="robodojo_${data_setting}"
data_source_path="${DEXBOTIC_ROOT}/dexbotic/data/data_source/robodojo_${data_setting}.py"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
base_model="${DM0_BASE_MODEL:-${DEXBOTIC_ROOT}/checkpoints/DM0-base}"

if [[ ! -f "${converted_data_root}/episode_000000.jsonl" ]]; then
  first_jsonl="$(find "${converted_data_root}" -maxdepth 1 -name 'episode_*.jsonl' | head -n 1 || true)"
  if [[ -z "${first_jsonl}" ]]; then
    echo "Converted Dexdata not found under ${converted_data_root}" >&2
    echo "Run process_data.sh first." >&2
    exit 1
  fi
fi

if [[ ! -f "${data_source_path}" ]]; then
  echo "Data source registration not found: ${data_source_path}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${base_model}/config.json" ]]; then
  echo "DM0 base model not found: ${base_model}" >&2
  echo "Download DM0-base with: hf download Dexmal/DM0-base --local-dir ${base_model}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"

if [[ -z "${DM0_GRAD_ACCUM:-}" ]]; then
  denom=$((DM0_BATCH_SIZE * NUM_GPUS))
  if (( DM0_GLOBAL_BATCH_SIZE % denom != 0 )); then
    echo "Cannot reach DM0_GLOBAL_BATCH_SIZE=${DM0_GLOBAL_BATCH_SIZE} with batch=${DM0_BATCH_SIZE} and num_gpus=${NUM_GPUS}." >&2
    echo "Set DM0_GRAD_ACCUM manually or adjust DM0_BATCH_SIZE / gpu_id." >&2
    exit 1
  fi
  export DM0_GRAD_ACCUM=$((DM0_GLOBAL_BATCH_SIZE / denom))
elif (( DM0_BATCH_SIZE * NUM_GPUS * DM0_GRAD_ACCUM != DM0_GLOBAL_BATCH_SIZE )); then
  effective_global=$((DM0_BATCH_SIZE * NUM_GPUS * DM0_GRAD_ACCUM))
  echo "Batch config mismatch: DM0_GLOBAL_BATCH_SIZE=${DM0_GLOBAL_BATCH_SIZE}, but batch*gpus*accum=${effective_global}." >&2
  exit 1
fi

export DM0_BENCH_NAME="${bench_name_registered}"
export DM0_DATASET_NAME="${bench_name_registered}"
export DM0_OUTPUT_DIR="${ckpt_dir}"
export DM0_MODEL_PATH="${base_model}"
export DM0_SEED="${seed}"

mkdir -p "${ckpt_dir}"

echo "[Dexbotic_DM0] converted_data_root=${converted_data_root}"
echo "[Dexbotic_DM0] bench_name=${bench_name_registered}"
echo "[Dexbotic_DM0] base_model=${base_model}"
echo "[Dexbotic_DM0] checkpoint_dir=${ckpt_dir}"
echo "[Dexbotic_DM0] seed=${seed}"
echo "[Dexbotic_DM0] gpu_id=${gpu_id}"
echo "[Dexbotic_DM0] num_gpus=${NUM_GPUS}"
echo "[Dexbotic_DM0] per_device_batch=${DM0_BATCH_SIZE}"
echo "[Dexbotic_DM0] grad_accum=${DM0_GRAD_ACCUM}"
echo "[Dexbotic_DM0] global_batch=$((DM0_BATCH_SIZE * NUM_GPUS * DM0_GRAD_ACCUM))"
echo "[Dexbotic_DM0] max_steps=${DM0_MAX_STEPS:-40000}"

cd "${DEXBOTIC_ROOT}"

torchrun --nproc_per_node="${NUM_GPUS}" "${EXP_SCRIPT}" \
  --task train \
  ${DM0_TRAIN_BACKEND:+--train-backend "${DM0_TRAIN_BACKEND}"}
