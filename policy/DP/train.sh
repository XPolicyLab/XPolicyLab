#!/bin/bash

dataset_name=${1}
ckpt_name=${2} # task_name
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}

DEBUG=False

addition_info=train
exp_name=${ckpt_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Get Action Dimension from env_cfg_type
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"

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
    bash process_data.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type}
fi

python train.py --config-name="${alg_name}.yaml" \
                dataset_name="${dataset_name}" \
                task.name="${ckpt_name}" \
                "task.shape_meta.action.shape=[${action_dim}]" \
                "task.shape_meta.obs.agent_pos.shape=[${action_dim}]" \
                task.dataset.zarr_path="data/${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}.zarr" \
                training.debug=$DEBUG \
                training.seed=${seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                setting=${env_cfg_type} \
                expert_data_num=${expert_data_num}