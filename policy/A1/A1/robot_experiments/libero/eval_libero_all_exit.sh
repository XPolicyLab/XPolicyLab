#!/bin/bash
export PYTHONPATH="/home/xurongtao/zhangjian/A1/team_code/A1:$PYTHONPATH"

export DATA_DIR=/mnt/data3/zhangjian/molmo/data
export HF_HOME=/mnt/data3/zhangjian/hf_cache

export CUDA_VISIBLE_DEVICES=1

source ~/.bashrc
conda activate a1

# 定义任务数组
TASKS=("libero_spatial" "libero_object" "libero_goal" "libero_10")
# TASKS=( "libero_object" "libero_goal" "libero_10")

# checkpoint_path="/mnt/data3/zhangjian/hf_cache/libero-4_extra_10_task_8_MolmoE-7B-10131629-5000_openai_seq368_flow_matching-qwen2_early_exit_two_images_proprio-8_ft_ah_fullyft_llm_bs176/step24000-unsharded"
checkpoint_path="/mnt/data3/zhangjian/hf_cache/hub/models--JianZhangAI--trained_model_early_exit/snapshots/250746ac29c8850549a95f9565d7004f0c19a2af/libero_4_molmo-7b-09242207_clip_l1_regression_early_exit_wrist_proprio_ft_ah_fullyft_llm_bs224/step27000-unsharded"

# 遍历任务并执行脚本
for TASK in "${TASKS[@]}"; do
    echo "Running task: $TASK"
    python robot_experiments/libero/eval_libero_early_exit.py \
        --task_suite_name "$TASK" \
        --exit_ratio 1.0 \
        --num_trials_per_task 50 \
        --action_head_flow_matching_inference_steps 10 \
        --device_batch_size 20 \
        --pretrained_checkpoint "$checkpoint_path" \
        --seed 6194
done