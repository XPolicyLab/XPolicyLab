#!/bin/bash

# == keep unchanged ==
policy_name=ACT
task_name=${1}
env_cfg=${2}
expert_data_num=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
# temporal_agg=${5} # use temporal_agg
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

action_dim=$(python3 -c '
import sys, os, json, yaml
env_cfg = yaml.safe_load(open(os.path.join("../../env_cfg", f"{sys.argv[1]}.yml"), "r", encoding="utf-8"))
robot_name = env_cfg["config"]["robot"]
robot_action_dim_info = json.load(open(os.path.join("../../env_cfg/robot", "_robot_info.json"), "r", encoding="utf-8"))[robot_name]
print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "$env_cfg")

export ACT_ACTION_DIM=${action_dim}

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --env_cfg ${env_cfg} \
    --ckpt_dir policy/ACT/act_ckpt/act-${task_name}/${env_cfg}-${expert_data_num}-${action_type} \
    --seed ${seed}