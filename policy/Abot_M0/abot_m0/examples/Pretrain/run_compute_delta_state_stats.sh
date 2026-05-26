python compute_delta_state_stats.py \
  --root_dir /path/to/dataset \  # path to single dataset, e.g. Robocoin, Agibot-Beta
  --input_euler_convention XYZ \
  --input_quaternion_order xyzw \
  --stats_file_policy overwrite \
  --max_workers 4 \
  --show_inner_pbar