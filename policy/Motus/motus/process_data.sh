export CUDA_VISIBLE_DEVICES=4

python data/lerobot/add_t5_cache_to_lerobot_dataset.py \
	--repo_id sim_stack_bowls_v21 \
	--root "${MOTUS_DATASET_ROOT:-/path/to/lerobot/sim_stack_bowls_v21}" \
	--wan_path "${MOTUS_WAN_PATH:-/path/to/model_weights/}" \
	--device cuda \
	--t5_folder_name t5_embedding