#!/usr/bin/env bash
# 8-task EventVLA batch evaluation config for the legacy starVLA Qwen3OFT
# raw_anchors_only checkpoint.
#
# Usage:
#   bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
#     examples/RoboTwin-Mem/eval_batch/weights_8tasks_raw_anchors_only_qwen3oft_0423.sh

TASK_CONFIG="demo_clean"
SEED=0
HOST="127.0.0.1"
UNNORM_KEY="new_embodiment"
ACTION_MODE="abs"
INSTRUCTION_TYPE="unseen"
USE_BF16=1

SERVER_READY_TIMEOUT=600
SERVER_START_MAX_RETRIES=2
SERVER_RETRY_SLEEP=20
SERVER_START_STAGGER_SECONDS=8
SERVER_STARTUP_STALL_TIMEOUT=300
AUTO_KILL_PORT_OCCUPY=1

EVENTVLA_PYTHON="/shared/smartbot/yangganlin/anaconda3/envs/starVLA/bin/python"
ROBOTWIN_MEM_PYTHON="/shared/smartbot/yangganlin/anaconda3/envs/RMBench/bin/python"
ROBOTWIN_MEM_ROOT="/mnt/workspace/yangganlin/tzz_workspace/final/RoboTwin-Mem"

POLICY_NAME="model2robotwin_mem_interface"

TASKS=(
  "rearrange_blocks" "put_back_block" "swap_T" "battery_try"
  "blocks_ranking_try" "cover_blocks" "swap_blocks" "press_button"
)

GPU_SLOTS=(4 4 5 5 6 6 7 7)
PORT_SLOTS=(5802 5803 5804 5805 5806 5807 5808 5809)

RAW_ANCHORS_ONLY_CKPT="/nav-oss/yangganlin/models/starVLA_0422_combined/0423_rmbench_qwen3OFT_raw_anchors_only/checkpoints/steps_100000_pytorch_model.pt"

WEIGHT_TASK1="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK2="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK3="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK4="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK5="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK6="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK7="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK8="$RAW_ANCHORS_ONLY_CKPT"

CKPT_SETTING_TASK1="raw_anchors_only"
CKPT_SETTING_TASK2="raw_anchors_only"
CKPT_SETTING_TASK3="raw_anchors_only"
CKPT_SETTING_TASK4="raw_anchors_only"
CKPT_SETTING_TASK5="raw_anchors_only"
CKPT_SETTING_TASK6="raw_anchors_only"
CKPT_SETTING_TASK7="raw_anchors_only"
CKPT_SETTING_TASK8="raw_anchors_only"
