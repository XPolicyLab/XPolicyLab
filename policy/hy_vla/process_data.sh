#!/usr/bin/env bash
set -euo pipefail

# Data processing for hy_vla (RoboDojo).
#
# Hy-VLA trains directly on RoboDojo HDF5 episodes; there is no bespoke
# XPolicyLab converter. This wrapper computes the normalization statistics
# (norm_stats.pkl) that both training and the eval-time policy server consume.
#
# The RoboDojo norm-stats computer scans the HDF5 tree
# ({hdf5_dir}/{task}/{robot}/data/episode_*.hdf5) directly -- no manifest CSV.
# `--umi-coord-frame` is REQUIRED: the training config (robodojo_hdf5.yaml) sets
# umi_coord_frame=True and the eval-time model normalizes in the UMI frame, so
# the stats must be UMI-frame too. Defaults mirror robodojo_hdf5.yaml
# (downsample_rate=1, chunk_size=25).
#
# Usage:
#   bash process_data.sh <hdf5_dir> <output_pkl> [downsample_rate] [chunk_size]
#
# See the Hy-Embodied repo for full data-collection + conversion docs:
#   https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hdf5_dir> <output_pkl> [downsample_rate] [chunk_size]" >&2
  exit 1
fi

hdf5_dir=$1
output_pkl=$2
downsample_rate=${3:-1}
chunk_size=${4:-25}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_ROOT="${HY_VLA_ROOT:-${POLICY_DIR}/Hy-Embodied-0.5-VLA}"

if [[ ! -d "${HY_VLA_ROOT}" ]]; then
  echo "[hy_vla] Hy-Embodied source not found at ${HY_VLA_ROOT}. Run install.sh first." >&2
  exit 1
fi

cd "${HY_VLA_ROOT}"
echo "[hy_vla] computing RoboDojo norm stats -> ${output_pkl}"
uv run python scripts/compute_norm_robodojo.py \
  --hdf5-dir "${hdf5_dir}" \
  --output "${output_pkl}" \
  --downsample-rate "${downsample_rate}" \
  --chunk-size "${chunk_size}" \
  --umi-coord-frame
