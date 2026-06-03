#!/bin/bash


config_name=${1}
ckpt_dir=${2}

CUDA_VISIBLE_DEVICES=0 python examples/label_frame_value.py --config_name ${config_name} --ckpt_dir ${ckpt_dir} --split all --no-with_vis
