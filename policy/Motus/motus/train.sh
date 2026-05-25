CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \

SEED=${SEED:-0}

torchrun \
	--nnodes=1 \
	--nproc_per_node=8 \
	--node_rank=0 \
	--master_addr=127.0.0.1 \
	--master_port=29500 \
	train/train.py \
	--deepspeed configs/zero2_stage2.json \
	--config configs/lerobot_RoboDojo_sim.yaml \
	--seed ${SEED} \
	--checkpoint_dir /mnt/xspark-data/xspark_shared/motus_ckpt/ \
	--run_name robodojo_sim_motus \
	--report_to tensorboard