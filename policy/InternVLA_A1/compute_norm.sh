REPO_ID=${1:-RoboDojo_sim_arx-x5_v30}

python internvla_a1/util_scripts/compute_norm_stats_single.py \
  --action_mode delta \
  --chunk_size 50 \
  --repo_id ${REPO_ID}