#!/usr/bin/env bash
# 8-task EventVLA batch evaluation config for the legacy starVLA QwenOFT
# pure_image_keyframe_memory checkpoint.
#
# Usage:
#   bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
#     examples/RoboTwin-Mem/eval_batch/weights_8tasks_pure_image_keyframe_memory_teacher_qwenoft.sh

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
  "cover_blocks_hard"
  "pick_the_unhidden_block"
  "find_seal_and_seal_stamp"
  "pick_objects_in_order"
  "put_back_block_hard"
  "press_button_keyframe"
  "rearrange_blocks_hard"
  "reproduce_route"
)

GPU_SLOTS=(0 0 1 1 2 2 3 3)
PORT_SLOTS=(5902 5903 5904 5905 5906 5907 5908 5909)

RMBENCH_HARD8_CKPT="/nav-oss/yangganlin/models/starVLA_0426_combined/20260608_rmbench_hard8_pure_image_keyframe_memory_teacher_qwenoft/checkpoints/steps_100000_pytorch_model.pt"

WEIGHT_TASK1="$RMBENCH_HARD8_CKPT"
WEIGHT_TASK2="$RMBENCH_HARD8_CKPT"
WEIGHT_TASK3="$RMBENCH_HARD8_CKPT"
WEIGHT_TASK4="$RMBENCH_HARD8_CKPT"
WEIGHT_TASK5="$RMBENCH_HARD8_CKPT"
WEIGHT_TASK6="$RMBENCH_HARD8_CKPT"
WEIGHT_TASK7="$RMBENCH_HARD8_CKPT"
WEIGHT_TASK8="$RMBENCH_HARD8_CKPT"

CKPT_SETTING_TASK1="pure_image_keyframe_memory"
CKPT_SETTING_TASK2="pure_image_keyframe_memory"
CKPT_SETTING_TASK3="pure_image_keyframe_memory"
CKPT_SETTING_TASK4="pure_image_keyframe_memory"
CKPT_SETTING_TASK5="pure_image_keyframe_memory"
CKPT_SETTING_TASK6="pure_image_keyframe_memory"
CKPT_SETTING_TASK7="pure_image_keyframe_memory"
CKPT_SETTING_TASK8="pure_image_keyframe_memory"
