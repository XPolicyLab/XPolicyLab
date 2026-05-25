export CUDA_VISIBLE_DEVICES=4

python data/lerobot/add_t5_cache_to_lerobot_dataset.py \
	--repo_id sim_stack_bowls_v21 \
	--root /vepfs-cnbje63de6fae220/xspark_shared/lerobot/sim_stack_bowls_v21 \
	--wan_path /vepfs-cnbje63de6fae220/xspark_shared/model_weights/ \
	--device cuda \
	--t5_folder_name t5_embedding