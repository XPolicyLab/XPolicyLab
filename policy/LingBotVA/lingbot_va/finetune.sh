export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPU=4 CONFIG_NAME='robotwin30_train' bash script/run_va_posttrain.sh