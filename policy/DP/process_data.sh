#!/bin/bash

task_name=${1}
env_cfg_type=${2}
expert_data_num=${3}
action_type=${4}

python diffusion_policy/process_data.py $task_name $env_cfg_type $expert_data_num $action_type