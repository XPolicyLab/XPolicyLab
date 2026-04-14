CONFIG_PATH=${1}
DATA_PATH=${2}
OUTPUT_PATH=${3}
GPU_ID=${4}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

bash train.sh tasks/vla/train_lingbotvla.py ${CONFIG_PATH} --data.train_path ${DATA_PATH} --train.output_dir ${OUTPUT_PATH}