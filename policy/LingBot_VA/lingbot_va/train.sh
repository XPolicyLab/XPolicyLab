export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l | xargs)

CONFIG_NAME='robotwin30_train' bash script/run_va_posttrain.sh
