#!/bin/bash
set -euo pipefail

# Mem_0 unified training: Execution Module, Planning Module (Mn), or both in sequence.
#
# Usage:
#   bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> \
#                 <action_type> <seed> <gpu_ids> [train_module]
#
# train_module (8th arg, default both):
#   execution  — torchrun + train_low.py (Qwen3-VL-2B)
#   planning   — LLaMA-Factory LoRA SFT + merge (Qwen3-VL-8B; Mn data)
#   both       — execution first, then planning (Mn full pipeline)
#
# M1 (single-stage) must pass execution explicitly:
#   bash train.sh RoboDojo test_data arx_x5 3 joint 42 0 execution
#
# Mn examples:
#   bash process_data.sh RoboDojo cover_blocks arx_x5 50 joint Mn
#   bash train.sh RoboDojo cover_blocks arx_x5 50 joint 42 0,1,2,3,4,5,6,7
#   bash train.sh RoboDojo cover_blocks arx_x5 50 joint 42 0,1,2,3,4,5,6,7 planning
#
# Execution tunables (env): BATCH_SIZE, TRAIN_STEPS, ENABLE_WANDB, IS_DEBUG,
#   NORM_STATS_PATH, MASTER_PORT, REPO_ID, ALLOW_NO_QWEN
# Planning tunables (env): STEPS, EPISODE_START_ID, EPISODE_END_ID (default: all
#   episodes in meta/info.json; use explicit range for Mn-only cotrain slices),
#   NUM_TRAIN_EPOCHS,
#   PER_DEVICE_TRAIN_BATCH_SIZE, MAX_SAMPLES, LEARNING_RATE, CUTOFF_LEN,
#   IMAGE_MAX_PIXELS, VIDEO_MAX_PIXELS, LLAMAFACTORY_ROOT, EXPORT_DIR,
#   CONDA_ENV_MEM0, CONDA_ENV_LLAMAFACTORY, ALLOW_NO_QWEN8B, FORCE_PREPARE,
#   FULL_DETERMINISM
# Shared: DRY_RUN (skip torchrun / LLaMA-Factory train+export; still writes configs)

usage() {
  echo "usage: bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> \\" >&2
  echo "                   <action_type> <seed> <gpu_ids> [train_module]" >&2
  echo "  train_module: execution | planning | both  (default: both)" >&2
}

dataset_name=${1:-}
ckpt_name=${2:-}
env_cfg_type=${3:-}
expert_data_num=${4:-}
action_type=${5:-}
seed=${6:-}
gpu_ids=${7:-}
train_module=${8:-}

if [[ -z "${dataset_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${expert_data_num}" \
      || -z "${action_type}" || -z "${seed}" || -z "${gpu_ids}" ]]; then
  usage
  exit 2
fi

if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
  echo -e "\033[31m[train] seed must be a non-negative integer, got: ${seed}\033[0m" >&2
  exit 1
fi

case "${train_module}" in
  execution|planning|both) ;;
  *)
    echo -e "\033[31m[train] invalid train_module: ${train_module}\033[0m" >&2
    usage
    exit 1
    ;;
esac

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${POLICY_DIR}/Mem_0"
ADAPTER_DIR="${UPSTREAM_DIR}/xpolicylab_adapter"
ORCHESTRATOR="${ADAPTER_DIR}/run_planning_train.py"

source "${ADAPTER_DIR}/_artifact_paths.sh"

run_id="$(mem0_ckpt_run_id "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
repo_id="${REPO_ID:-$(mem0_resolve_dataset_dir "${POLICY_DIR}" "${dataset_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${expert_data_num}")}"

run_execution_train() {
  local ckpt_dir="${POLICY_DIR}/checkpoints/${run_id}"
  local gen_config="${ckpt_dir}/train_config.yaml"
  local batch_size=${BATCH_SIZE:-56}
  local train_steps=${TRAIN_STEPS:-100000}
  local enable_wandb=${ENABLE_WANDB:-true}
  local is_debug=${IS_DEBUG:-false}
  local master_port=${MASTER_PORT:-29500}
  local dry_run=${DRY_RUN:-false}

  if [[ ! -d "${repo_id}" ]]; then
    echo -e "\033[31m[train:execution] LeRobot dataset not found: ${repo_id}\033[0m" >&2
    echo "Run: bash process_data.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} <M1|Mn>   (or set REPO_ID=...)" >&2
    exit 1
  fi

  local qwen_dir="${UPSTREAM_DIR}/checkpoints/Qwen3-VL-2B-Instruct"
  if [[ ! -d "${qwen_dir}" && "${ALLOW_NO_QWEN:-false}" != "true" ]]; then
    echo -e "\033[31m[train:execution] Qwen3-VL-2B backbone not found: ${qwen_dir}\033[0m" >&2
    echo "Download it (cd Mem_0/checkpoints && python _download.py), or set ALLOW_NO_QWEN=true for a smoke run." >&2
    exit 1
  fi

  IFS=',' read -ra _gpu_arr <<< "${gpu_ids}"
  local nproc=${#_gpu_arr[@]}

  local norm_args=()
  [[ -n "${NORM_STATS_PATH:-}" ]] && norm_args+=( --norm_stats_path "${NORM_STATS_PATH}" )

  mkdir -p "${ckpt_dir}"
  python "${ADAPTER_DIR}/gen_train_config.py" \
      --repo_id "${repo_id}" \
      --checkpoint_dir "${ckpt_dir}" \
      --wandb_run_name "${run_id}" \
      --out "${gen_config}" \
      --seed "${seed}" \
      --batch_size "${batch_size}" \
      --train_steps "${train_steps}" \
      --enable_wandb "${enable_wandb}" \
      --is_debug "${is_debug}" \
      "${norm_args[@]}"

  if [[ "${dry_run}" == "true" ]]; then
    echo -e "\033[33m[train:execution] DRY_RUN: wrote ${gen_config}; skipping torchrun\033[0m"
    return 0
  fi

  echo -e "\033[33m[train:execution] GPUs=${gpu_ids} (nproc_per_node=${nproc}); run=${run_id}\033[0m"
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
}

run_planning_train() {
  local lf_root="${LLAMAFACTORY_ROOT:-${UPSTREAM_DIR}/LlamaFactory}"
  local base_output_dir="${POLICY_DIR}/checkpoints"
  local run_config_dir="${base_output_dir}/${run_id}"
  local adapter_dir="${base_output_dir}/${run_id}_planning_sft_lora"
  local merged_dir="${base_output_dir}/${run_id}_planning_merged"

  local episode_start=${EPISODE_START_ID:-0}
  local steps="${STEPS:-prepare copy train merge}"
  local enable_wandb=${ENABLE_WANDB:-true}
  local dry_run=${DRY_RUN:-false}
  local force_prepare=${FORCE_PREPARE:-false}
  local full_determinism=${FULL_DETERMINISM:-false}

  local conda_env_mem0=${CONDA_ENV_MEM0:-mem0}
  local conda_env_llama=${CONDA_ENV_LLAMAFACTORY:-llama_factory}

  local num_epochs=${NUM_TRAIN_EPOCHS:-25}
  local batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-16}
  local max_samples=${MAX_SAMPLES:-1000}
  local learning_rate=${LEARNING_RATE:-1.0e-4}
  local cutoff_len=${CUTOFF_LEN:-4096}
  local image_max_pixels=${IMAGE_MAX_PIXELS:-131072}
  local video_max_pixels=${VIDEO_MAX_PIXELS:-16384}
  local prep_workers=${PREPROCESSING_NUM_WORKERS:-16}
  local dl_workers=${DATALOADER_NUM_WORKERS:-8}

  if [[ ! -d "${repo_id}" ]]; then
    echo -e "\033[31m[train:planning] LeRobot dataset not found: ${repo_id}\033[0m" >&2
    echo "Run: bash process_data.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} Mn" >&2
    exit 1
  fi

  local meta_json="${repo_id}/meta/info.json"
  if [[ ! -f "${meta_json}" ]]; then
    echo -e "\033[31m[train:planning] Missing meta/info.json under ${repo_id}\033[0m" >&2
    exit 1
  fi

  local read_meta_py
  read_meta_py=$(python3 - "${meta_json}" "${episode_start}" "${EPISODE_END_ID:-}" <<'PY'
import json, sys

meta_path = sys.argv[1]
episode_start = int(sys.argv[2])
explicit_end = sys.argv[3].strip()

meta = json.load(open(meta_path))
total = int(meta.get("total_episodes", 0))
features = meta.get("features") or {}

if total <= 0:
    print("INVALID_TOTAL")
    sys.exit(0)
if "subtask_end" not in features:
    print("MISSING_SUBTASK_END")
    sys.exit(0)

if explicit_end:
    episode_end = int(explicit_end)
else:
    episode_end = total

if episode_start < 0 or episode_start >= total:
    print(f"INVALID_START {episode_start} {total}")
    sys.exit(0)
if episode_end <= episode_start:
    print(f"INVALID_RANGE {episode_start} {episode_end}")
    sys.exit(0)
if episode_end > total:
    print(f"EPISODE_OVERFLOW {total} {episode_end}")
    sys.exit(0)

print(f"OK {episode_end} {total}")
PY
)

  if [[ "${read_meta_py}" == "MISSING_SUBTASK_END" ]]; then
    echo -e "\033[31m[train:planning] Dataset lacks subtask_end (not Mn-style). Use process_data.sh ... Mn\033[0m" >&2
    exit 1
  fi
  if [[ "${read_meta_py}" == "INVALID_TOTAL" ]]; then
    echo -e "\033[31m[train:planning] total_episodes is missing or zero in ${meta_json}\033[0m" >&2
    exit 1
  fi
  if [[ "${read_meta_py}" == INVALID_START* ]]; then
    read -r _ bad_start total_eps <<< "${read_meta_py#INVALID_START }"
    echo -e "\033[31m[train:planning] EPISODE_START_ID=${bad_start} out of range [0, $((total_eps - 1))]\033[0m" >&2
    exit 1
  fi
  if [[ "${read_meta_py}" == INVALID_RANGE* ]]; then
    read -r _ bad_start bad_end <<< "${read_meta_py#INVALID_RANGE }"
    echo -e "\033[31m[train:planning] EPISODE_END_ID=${bad_end} must be > EPISODE_START_ID=${bad_start}\033[0m" >&2
    exit 1
  fi
  if [[ "${read_meta_py}" == EPISODE_OVERFLOW* ]]; then
    read -r _ total_eps bad_end <<< "${read_meta_py#EPISODE_OVERFLOW }"
    echo -e "\033[31m[train:planning] EPISODE_END_ID=${bad_end} exceeds total_episodes=${total_eps} in ${meta_json}\033[0m" >&2
    exit 1
  fi
  if [[ "${read_meta_py}" != OK* ]]; then
    echo -e "\033[31m[train:planning] Failed to read dataset metadata: ${read_meta_py}\033[0m" >&2
    exit 1
  fi

  read -r _ episode_end total_eps <<< "${read_meta_py#OK }"
  if [[ -n "${EPISODE_END_ID:-}" ]]; then
    echo -e "\033[33m[train:planning] episodes [${episode_start}, ${episode_end}) of total_episodes=${total_eps} (EPISODE_END_ID set)\033[0m"
  else
    echo -e "\033[33m[train:planning] episodes [${episode_start}, ${episode_end}) of total_episodes=${total_eps} (auto from meta/info.json)\033[0m"
  fi

  local qwen8b_dir="${UPSTREAM_DIR}/checkpoints/Qwen3-VL-8B-Instruct"
  if [[ "${ALLOW_NO_QWEN8B:-false}" != "true" ]]; then
    shopt -s nullglob
    local safetensors=( "${qwen8b_dir}"/model*.safetensors )
    shopt -u nullglob
    if [[ ${#safetensors[@]} -eq 0 ]]; then
      echo -e "\033[31m[train:planning] Qwen3-VL-8B weights not found under ${qwen8b_dir}\033[0m" >&2
      echo "Download: cd Mem_0/checkpoints && python _download.py" >&2
      echo "Smoke only: ALLOW_NO_QWEN8B=true DRY_RUN=true bash train.sh ... planning" >&2
      exit 1
    fi
  fi

  if [[ ! -d "${lf_root}" ]]; then
    echo -e "\033[31m[train:planning] LLaMA-Factory root not found: ${lf_root}\033[0m" >&2
    echo "Run: bash install_planning.sh" >&2
    exit 1
  fi

  source "$(conda info --base)/etc/profile.d/conda.sh"
  for env_name in "${conda_env_mem0}" "${conda_env_llama}"; do
    if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
      echo -e "\033[31m[train:planning] Conda env not found: ${env_name}\033[0m" >&2
      [[ "${env_name}" == "${conda_env_llama}" ]] && echo "Run: bash install_planning.sh" >&2
      [[ "${env_name}" == "${conda_env_mem0}" ]] && echo "Run: bash install.sh mem0" >&2
      exit 1
    fi
  done

  echo -e "\033[33m[train:planning] run=${run_id} GPUs=${gpu_ids} steps=${steps}\033[0m"
  export CUDA_VISIBLE_DEVICES="${gpu_ids}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export TOKENIZERS_PARALLELISM=false
  export ENABLE_WANDB="${enable_wandb}"

  local orchestrator_args=(
    --lerobot_dataset_path "${repo_id}"
    --llamafactory_root "${lf_root}"
    --base_output_dir "${base_output_dir}"
    --run_name "${run_id}"
    --run_config_dir "${run_config_dir}"
    --adapter_output_dir "${adapter_dir}"
    --merged_output_dir "${merged_dir}"
    --seed "${seed}"
    --episode_start_id "${episode_start}"
    --episode_end_id "${episode_end}"
    --max_samples "${max_samples}"
    --num_train_epochs "${num_epochs}"
    --per_device_train_batch_size "${batch_size}"
    --learning_rate "${learning_rate}"
    --cutoff_len "${cutoff_len}"
    --image_max_pixels "${image_max_pixels}"
    --video_max_pixels "${video_max_pixels}"
    --preprocessing_num_workers "${prep_workers}"
    --dataloader_num_workers "${dl_workers}"
    --conda_env_mem0 "${conda_env_mem0}"
    --conda_env_llamafactory "${conda_env_llama}"
    --steps ${steps}
  )

  [[ -n "${EXPORT_DIR:-}" ]] && orchestrator_args+=( --export_dir "${EXPORT_DIR}" )
  [[ "${dry_run}" == "true" ]] && orchestrator_args+=( --dry-run )
  [[ "${force_prepare}" == "true" ]] && orchestrator_args+=( --force-prepare )
  [[ "${full_determinism}" == "true" ]] && orchestrator_args+=( --full-determinism )

  python3 "${ORCHESTRATOR}" "${orchestrator_args[@]}"
}

echo -e "\033[36m[train] module=${train_module} run=${run_id}\033[0m"

case "${train_module}" in
  execution) run_execution_train ;;
  planning)  run_planning_train ;;
  both)
    run_execution_train
    run_planning_train
    ;;
esac
