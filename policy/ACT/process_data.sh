#!/bin/bash

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

python detr/process_data.py $dataset_name $task_name $env_cfg_type $expert_data_num $action_type