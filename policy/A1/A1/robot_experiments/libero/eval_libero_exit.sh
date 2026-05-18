#!/bin/bash

# ============================================
# 加载个人环境配置
# ============================================
if [ -f "$PWD/.env.personal" ]; then
  echo "[env] 加载 .env.personal"
  source "$PWD/.env.personal"
fi

# ============================================
# 激活 Conda 环境
# ============================================
if [ -n "$CONDA_ROOT" ] && [ -n "$CONDA_ENV" ]; then
  echo "[conda] 从 $CONDA_ROOT 激活环境: $CONDA_ENV"
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

heckpoint_path="model/checkpoints/zj_libero_exit/libero-4_Molmo-7B-D-0924_openai_seq680_flow_matching-qwen2-new-pvf_early_exit_two_images_crop_overlap-and-resize-c2-4_proprio-8_ft_ah_fullyft_llm_bs96_lr1e-5/step92500-unsharded"

TASK="libero_object"
# TASKS=("libero_spatial" "libero_object" "libero_goal" "libero_10")

python robot_experiments/libero/eval_libero_early_exit.py \
    --task_suite_name "$TASK" \
    --exit_ratio 1.0 \
    --num_trials_per_task 30 \
    --action_head_flow_matching_inference_steps 10 \
    --device_batch_size 20 \
    --pretrained_checkpoint "$checkpoint_path" \
    --seed 6194