#!/bin/bash

# ============================================
# Load personal environment configuration
# ============================================
if [ -f "$PWD/.env.personal" ]; then
  echo "[env] Loading .env.personal"
  source "$PWD/.env.personal"
fi

# ============================================
# Activate Conda environment
# ============================================
if [ -n "$CONDA_ROOT" ] && [ -n "$CONDA_ENV" ]; then
  echo "[conda] Activating environment from $CONDA_ROOT: $CONDA_ENV"
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

checkpoint_path="model/libero_exit"

TASKS=("libero_spatial" "libero_object" "libero_goal" "libero_10")
for TASK in "${TASKS[@]}"; do
    python robot_experiments/libero/eval_libero_early_exit.py \
        --task_suite_name "$TASK" \
        --exit_ratio 1.0 \
        --num_trials_per_task 50 \
        --action_head_flow_matching_inference_steps 10 \
        --device_batch_size 20 \
        --pretrained_checkpoint "$checkpoint_path" \
        --seed 6194
done