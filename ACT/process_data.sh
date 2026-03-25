#!/bin/bash

task_name=${1}
env_cfg=${2}
expert_data_num=${3}

python detr/process_data.py $task_name $env_cfg $expert_data_num