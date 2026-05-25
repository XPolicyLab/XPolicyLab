DATA_PATH=${1}
WAN22_PATH=${2}

python -m scripts.compute_norm_stats \
  --data_paths "${DATA_PATH}" \
  --output_path "./norm_stats_delta.json" \
  --embodiment_id 0 \
  --delta-mask True True True True True True False True True True True True True False \
  --sample-rate 1.0 \
  --action-chunk 48 \
  --action-dim 16 \

python -m scripts.compute_t5_embedding \
  --repo_id "${DATA_PATH}" \
  --root "${DATA_PATH}" \
  --wan_path "${WAN22_PATH}" \
  --device "cuda" \
  --text_len 512 \
  --t5_folder_name "t5_embedding"