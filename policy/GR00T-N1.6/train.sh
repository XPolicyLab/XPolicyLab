#!/bin/bash
set -e

# ==================== 参数定义 ====================
dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
gpu_id=${6}
seed=${7}