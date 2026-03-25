#!/bin/bash

task_name=${1}
env_cfg=${2}
expert_data_num=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False
save_ckpt=True

addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Get Action Dimension from env_cfg
action_dim=$(python3 -c '
import sys, os, json, yaml
env_cfg = yaml.safe_load(open(os.path.join("../../env_cfg", f"{sys.argv[1]}.yml"), "r", encoding="utf-8"))
robot_name = env_cfg["config"]["robot"]
robot_action_dim_info = json.load(open(os.path.join("../../env_cfg/robot", "_robot_info.json"), "r", encoding="utf-8"))[robot_name]
print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "$env_cfg")

alg_name=robot_dp

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d  ]; then
    bash process_data.sh ${task_name} ${env_cfg} ${expert_data_num} ${action_type}
fi

python train.py --config-name="${alg_name}.yaml" \
                task.name="${task_name}" \
                "task.shape_meta.action.shape=[${action_dim}]" \
                "task.shape_meta.obs.agent_pos.shape=[${action_dim}]" \
                task.dataset.zarr_path="data/${task_name}-${env_cfg}-${expert_data_num}-${action_type}.zarr" \
                training.debug=$DEBUG \
                training.seed=${seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                setting=${env_cfg} \
                expert_data_num=${expert_data_num}
                # checkpoint.save_ckpt=${save_ckpt}
                # hydra.run.dir=${run_dir} \