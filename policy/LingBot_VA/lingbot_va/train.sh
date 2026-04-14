GPU_ID=${1}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

export NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l | xargs)

CONFIG_NAME='robotwin30_train' bash script/run_va_posttrain.sh
