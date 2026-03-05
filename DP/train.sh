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

data_path="./data/${task_name}-${env_cfg}-${expert_data_num}-${action_type}.zarr"

ZARR="${data_path}/data/action/.zarray"
action_dim=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["chunks"][1])' "$ZARR")

alg_name=robot_dp_$action_dim

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
    bash process_data.sh ${task_name} ${env_cfg} ${expert_data_num}
fi

python train.py --config-name=${alg_name}.yaml \
                task.name=${task_name} \
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