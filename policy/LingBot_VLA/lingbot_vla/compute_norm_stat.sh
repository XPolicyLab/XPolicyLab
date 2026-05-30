YAML_PATH=${1:-configs/norm/robodojo_sim_arx_x5_customized.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

torchrun --nnodes=1 --nproc-per-node=1 --node-rank=0   --master-addr=127.0.0.1 --master-port=62500   scripts/compute_norm_robotwin_5.py ${YAML_PATH}