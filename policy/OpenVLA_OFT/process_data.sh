#!/bin/bash

task_name=${1}
env_cfg_type=${2}
expert_data_num=${3}
action_type=${4}

python scripts/process_data.py \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --expert_data_num "${expert_data_num}" \
    --action_type "${action_type}" \
    --percent_val 0.05 \
    --img_resize_size 256

# --dataset_path path_to_original_data   --out_base_dir output_data_dir   --percent_val 0.05 --instruction_dir path_to_the_instructions
