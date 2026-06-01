#!/bin/bash

# * usage: ./vis_value.sh CONFIG_NAME CKPT_DIR


cd $(dirname $(realpath $0))

config_name=${1}
ckpt_dir=${2}
split=${3:-"all"}

CUDA_VISIBLE_DEVICES=0 python examples/custom_vis_torch.py --config_name ${config_name} --ckpt_dir ${ckpt_dir} --split ${split}
