#!/bin/bash
set -e

# ==================== 参数定义 ====================
dataset_name=$1 # requried
task_name=$2
ckpt_name=$3 # requried
env_cfg_type=$4 # requried
expert_data_num=$5 # requried
action_type=$6 # requried
seed=$7 # requried
gpu_id=$8 # requried