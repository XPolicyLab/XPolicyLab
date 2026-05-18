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

export VLA_CONFIG_YAML=libero_simulation.yaml

TASKS=("libero_spatial" "libero_object" "libero_goal" "libero_10")
checkpoint_dir=model/libero

for TASK in "${TASKS[@]}"; do
    echo "Running task: $TASK"
    python robot_experiments/libero/eval_libero.py \
        --task_suite_name "$TASK" \
        --pretrained_checkpoint $checkpoint_dir \
        --local_log_dir $checkpoint_dir/eval_logs \
        --save_rollout_video_path $checkpoint_dir
done
