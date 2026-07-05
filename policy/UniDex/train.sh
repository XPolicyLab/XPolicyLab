#!/bin/bash
set -e

# ==================== 参数定义 ====================
bench_name=$1 # required
ckpt_name=$2 # required
env_cfg_type=$3 # required
expert_data_num=$4 # required
action_type=$5 # required
seed=$6 # required
gpu_id=$7
