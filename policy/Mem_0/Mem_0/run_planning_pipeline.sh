#!/usr/bin/env bash
#
# Planning Module pipeline: data preparation -> copy to LLaMA-Factory -> train -> merge LoRA.
# Edit only the variables in the "User configuration" section below. Do not edit source code.
#
# Usage:
#   cd policy/Mem_0
#   ./run_planning_pipeline.sh
#
# Optional: run specific steps (prepare, copy, train, merge):
#   STEPS="copy train merge" ./run_planning_pipeline.sh
# Steps are auto-skipped if already done: prepare when JSON exists under llamafactory_data/;
# copy when the same JSON exists under LlamaFactory/data/.
#

set -e

# -----------------------------------------------------------------------------
# User configuration: edit only this section (all values are env-overridable)
# -----------------------------------------------------------------------------
# Portable defaults derived from this script's location (this file lives at the
# Mem_0 upstream root). Override any value via the environment instead of editing.
_PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LeRobot dataset path (required). e.g. ${_PIPELINE_DIR}/lerobot_datasets/<name>
LEROBOT_DATASET_PATH="${LEROBOT_DATASET_PATH:-}"

# Episode range for data preparation (inclusive start, exclusive end)
EPISODE_START_ID=${EPISODE_START_ID:-0}
EPISODE_END_ID=${EPISODE_END_ID:-50}

# LLaMA-Factory repository root. Defaults to the vendored sub-directory.
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-${_PIPELINE_DIR}/LlamaFactory}"

# Base directory for LoRA output and merged model. Script creates {dataset_name}_sft_lora under it.
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${_PIPELINE_DIR}/checkpoints}"

# Merged model output directory (optional). If empty, uses BASE_OUTPUT_DIR/Qwen3-VL-8B-Instruct-{dataset_name}
EXPORT_DIR=""

# Training options (optional; change if needed)
MAX_SAMPLES=1000
NUM_TRAIN_EPOCHS=25
PER_DEVICE_TRAIN_BATCH_SIZE=16
LEARNING_RATE="1.0e-4"
REPORT_TO="wandb"
CUTOFF_LEN=4096
IMAGE_MAX_PIXELS=131072
VIDEO_MAX_PIXELS=16384
PREPROCESSING_NUM_WORKERS=16
DATALOADER_NUM_WORKERS=8

# Merge options (optional)
EXPORT_SIZE=5
EXPORT_DEVICE="cpu"

# Conda environments: data prep uses CONDA_ENV_MEM0; train/merge use CONDA_ENV_LLAMAFACTORY
CONDA_ENV_MEM0="mem0"
CONDA_ENV_LLAMAFACTORY="llama_factory"

# Steps to run: prepare, copy, train, merge. Default: all. Example: STEPS="copy train merge"
STEPS="${STEPS:-prepare copy train merge}"

# -----------------------------------------------------------------------------
# Paths (do not edit unless you move the script)
# -----------------------------------------------------------------------------
MEM0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MEM0_DIR"
SCRIPTS_DIR="${MEM0_DIR}/scripts"
DATA_PREP_SCRIPT="${SCRIPTS_DIR}/llama_data_preparation/llamafactory_data_preparation.py"
PIPELINE_SCRIPT="${SCRIPTS_DIR}/planning_train_pipeline.py"

# -----------------------------------------------------------------------------
# Progress bar helpers
# -----------------------------------------------------------------------------
# Usage: print_step_progress <current> <total> <step_name>
print_step_progress() {
  local current=$1
  local total=$2
  local name=$3
  local bar_width=24
  local filled=$((current * bar_width / total))
  [[ $filled -gt $bar_width ]] && filled=$bar_width
  local empty=$((bar_width - filled))
  local filled_chars
  local empty_chars
  filled_chars=$(printf '%*s' "$filled" '' | tr ' ' '=')
  empty_chars=$(printf '%*s' "$empty" '' | tr ' ' '-')
  printf "\n"
  printf "======================================================================\n"
  printf "  [%s%s] Step %d/%d: %s\n" "$filled_chars" "$empty_chars" "$current" "$total" "$name"
  printf "======================================================================\n"
  printf "\n"
}

# Count total steps that will run (prepare = 1 step, copy+train+merge = 1 step)
count_steps() {
  local n=0
  [[ " $STEPS " == *" prepare "* ]] && ((n++))
  [[ " $STEPS " == *" copy "* || " $STEPS " == *" train "* || " $STEPS " == *" merge "* ]] && ((n++))
  [[ $n -eq 0 ]] && n=1
  echo $n
}

# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
if [[ -z "$LEROBOT_DATASET_PATH" ]]; then
  echo "Error: LEROBOT_DATASET_PATH is not set. Edit the 'User configuration' section in this script." >&2
  exit 1
fi
if [[ ! -d "$LEROBOT_DATASET_PATH" ]]; then
  echo "Error: LEROBOT_DATASET_PATH is not a directory: $LEROBOT_DATASET_PATH" >&2
  exit 1
fi
if [[ -z "$LLAMAFACTORY_ROOT" ]]; then
  echo "Error: LLAMAFACTORY_ROOT is not set. Edit the 'User configuration' section in this script." >&2
  exit 1
fi
if [[ ! -d "$LLAMAFACTORY_ROOT" ]]; then
  echo "Error: LLAMAFACTORY_ROOT is not a directory: $LLAMAFACTORY_ROOT" >&2
  exit 1
fi
if [[ -z "$BASE_OUTPUT_DIR" ]]; then
  echo "Error: BASE_OUTPUT_DIR is not set. Edit the 'User configuration' section in this script." >&2
  exit 1
fi

# Dataset name (from LeRobot path) for skip detection
DATASET_NAME=$(basename "${LEROBOT_DATASET_PATH%/}")

TOTAL_STEPS=$(count_steps)
CURRENT_STEP=0

# -----------------------------------------------------------------------------
# Step 1: Data preparation (runs in mem0 env). Skip if output already exists.
# -----------------------------------------------------------------------------
if [[ " $STEPS " == *" prepare "* ]]; then
  PREPARE_JSON="${MEM0_DIR}/llamafactory_data/${DATASET_NAME}/${DATASET_NAME}_high_level_finetune_data.json"
  if [[ -f "$PREPARE_JSON" ]]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_step_progress "$CURRENT_STEP" "$TOTAL_STEPS" "Data preparation (already done, skipping)"
    echo "[Step 1] Output already exists: $PREPARE_JSON — skipping."
  else
    CURRENT_STEP=$((CURRENT_STEP + 1))
    EPISODE_COUNT=$((EPISODE_END_ID - EPISODE_START_ID))
    print_step_progress "$CURRENT_STEP" "$TOTAL_STEPS" "Data preparation [episodes $EPISODE_START_ID~$((EPISODE_END_ID - 1)), $EPISODE_COUNT total] (conda: $CONDA_ENV_MEM0)"
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV_MEM0" python -u "$DATA_PREP_SCRIPT" \
      --lerobot_dataset_path "$LEROBOT_DATASET_PATH" \
      --episode_start_id "$EPISODE_START_ID" \
      --episode_end_id "$EPISODE_END_ID"
    echo "[Step 1] Done."
  fi
fi

# -----------------------------------------------------------------------------
# Steps 2–4: Copy, train, merge (via Python pipeline script; train/merge use llama_factory env)
# Skip copy if data already present in LLaMA-Factory.
# -----------------------------------------------------------------------------
RUN_STEPS=""
for s in copy train merge; do
  if [[ " $STEPS " == *" $s "* ]]; then
    RUN_STEPS="${RUN_STEPS} ${s}"
  fi
done
RUN_STEPS="${RUN_STEPS# }"

# If copy is requested but already done (JSON exists in LlamaFactory/data), drop copy from RUN_STEPS
COPY_JSON="${LLAMAFACTORY_ROOT}/data/${DATASET_NAME}_high_level_finetune_data.json"
if [[ -n "$RUN_STEPS" ]] && [[ " $RUN_STEPS " == *" copy "* ]] && [[ -f "$COPY_JSON" ]]; then
  echo "[Copy] Already done: $COPY_JSON exists — skipping copy step."
  RUN_STEPS=$(echo " $RUN_STEPS " | sed 's/ copy / /' | tr -s ' ' | xargs)
fi

if [[ -n "$RUN_STEPS" ]]; then
  CURRENT_STEP=$((CURRENT_STEP + 1))
  print_step_progress "$CURRENT_STEP" "$TOTAL_STEPS" "Copy to LLaMA-Factory, Train, Merge ($RUN_STEPS)"
  ARGS=(
    --lerobot_dataset_path "$LEROBOT_DATASET_PATH"
    --llamafactory_root "$LLAMAFACTORY_ROOT"
    --base_output_dir "$BASE_OUTPUT_DIR"
    --episode_start_id "$EPISODE_START_ID"
    --episode_end_id "$EPISODE_END_ID"
    --max_samples "$MAX_SAMPLES"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --report_to "$REPORT_TO"
    --cutoff_len "$CUTOFF_LEN"
    --image_max_pixels "$IMAGE_MAX_PIXELS"
    --video_max_pixels "$VIDEO_MAX_PIXELS"
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --export_size "$EXPORT_SIZE"
    --export_device "$EXPORT_DEVICE"
    --conda_env_mem0 "$CONDA_ENV_MEM0"
    --conda_env_llamafactory "$CONDA_ENV_LLAMAFACTORY"
    --steps $RUN_STEPS
  )
  if [[ -n "$EXPORT_DIR" ]]; then
    ARGS+=(--export_dir "$EXPORT_DIR")
  fi
  python "$PIPELINE_SCRIPT" "${ARGS[@]}"
  echo "[Steps 2–4] Done."
fi

echo ""
echo "========================================================================"
echo "  Pipeline finished."
echo "========================================================================"
