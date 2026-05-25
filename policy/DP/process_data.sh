#!/bin/bash

dataset_name=${1}
ckpt_name=${2} # task_name
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

python diffusion_policy/process_data.py $dataset_name $ckpt_name $env_cfg_type $expert_data_num $action_type