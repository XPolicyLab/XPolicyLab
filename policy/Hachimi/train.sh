#!/usr/bin/env bash
# Train the submitted checkpoint from scratch (pi05_base). 8x 80GB GPUs, ~76 h.
#
# Inputs (env vars, all required):
#   HF_LEROBOT_HOME   directory holding the LeRobot v2.1 datasets; must contain
#                     RoboDojo_lerobot_v21_merged (see process_data.sh)
#   PI05_BASE_PATH    pi05_base checkpoint dir (from gs://openpi-assets/checkpoints/pi05_base)
#   VGGT_WEIGHT_PATH  VGGT-1B weights dir (facebook/VGGT-1B)
#   PI05SF_ASSETS_DIR directory where compute_norm_stats wrote
#                     RoboDojo_lerobot_v21_merged/norm_stats.json (see process_data.sh)
#
# Notes on the two launch flags that differ from the stock Spatial Forcing recipe:
#   --no-sf-cache-enable        VGGT targets are computed online (no offline cache);
#                               REQUIRED for the merged dataset (the cache is readonly
#                               with miss_policy=error, so new episodes would fail at step 0).
#   --align-vggt-devices 4 --fsdp-devices 4
#                               with the cache disabled the 8 GPUs are split 4 (VGGT) / 4
#                               (policy FSDP); 4/4 is the only split that keeps the stock
#                               batch_size=256 divisible.
# Everything else (batch 256, lr schedule, align_loss_coeff 0.2, 60k steps) is the stock
# config pi05sfwam_jax_robodojo_v21_merged. Checkpoints save every 5000 steps; the final
# checkpoint is step 59999 (num_train_steps - 1).
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${POLICY_DIR}/open_sf_wam"

: "${HF_LEROBOT_HOME:?set HF_LEROBOT_HOME}"
: "${PI05_BASE_PATH:?set PI05_BASE_PATH}"
: "${VGGT_WEIGHT_PATH:?set VGGT_WEIGHT_PATH}"
: "${PI05SF_ASSETS_DIR:?set PI05SF_ASSETS_DIR}"
export HF_LEROBOT_HOME PI05_BASE_PATH VGGT_WEIGHT_PATH PI05SF_ASSETS_DIR

EXP_NAME="${EXP_NAME:-hachimi_v1}"
STEPS="${STEPS:-60000}"
WORKERS="${WORKERS:-32}"
CKPT_BASE="${CKPT_BASE:-${CODE_ROOT}/checkpoints_out}"

cd "${CODE_ROOT}"
exec .venv/bin/python scripts/train_align.py pi05sfwam_jax_robodojo_v21_merged \
  --exp-name "${EXP_NAME}" \
  --num-train-steps "${STEPS}" \
  --num-workers "${WORKERS}" \
  --save-interval 5000 \
  --no-sf-cache-enable \
  --align-vggt-devices 4 \
  --fsdp-devices 4 \
  --checkpoint-base-dir "${CKPT_BASE}"
