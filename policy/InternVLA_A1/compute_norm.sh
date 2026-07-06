#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${1:-RoboDojo_sim_arx-x5_v30}
ACTION_MODE=${INTERNVLA_ACTION_MODE:-delta}

python internvla_a1/util_scripts/compute_norm_stats_single.py \
  --action_mode "${ACTION_MODE}" \
  --chunk_size 50 \
  --repo_id "${REPO_ID}"