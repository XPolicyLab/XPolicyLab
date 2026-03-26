#!/bin/bash

task_name=${1}
env_cfg=${2}
expert_data_num=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}

# Get Action Dimension from env_cfg
action_dim=$(python3 -c '
import sys, os, json, yaml
env_cfg = yaml.safe_load(open(os.path.join("../../env_cfg", f"{sys.argv[1]}.yml"), "r", encoding="utf-8"))
robot_name = env_cfg["config"]["robot"]
robot_action_dim_info = json.load(open(os.path.join("../../env_cfg/robot", "_robot_info.json"), "r", encoding="utf-8"))[robot_name]
print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "$env_cfg")

export ACT_ACTION_DIM=${action_dim}

python3 imitate_episodes.py \
    --task_name ${task_name}-${env_cfg}-${expert_data_num}-${action_type} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${env_cfg}-${expert_data_num}-${action_type} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 6000 \
    --seed ${seed}
