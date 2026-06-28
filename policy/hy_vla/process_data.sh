#!/usr/bin/env bash
set -euo pipefail

# Data processing for hy_vla.
#
# Hy-VLA trains on the Hy-Embodied UMI / RoboTwin data pipeline that ships in
# the Hy-Embodied source tree, not on a bespoke XPolicyLab converter. This
# wrapper computes the normalization statistics (norm_stats.pkl) that the
# policy server consumes at eval time.
#
# Usage:
#   bash process_data.sh <manifest_csv> <hdf5_dir> <output_pkl> [downsample_rate] [chunk_size]
#
# See the Hy-Embodied repo for full data-collection + conversion docs:
#   https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <manifest_csv> <hdf5_dir> <output_pkl> [downsample_rate] [chunk_size]" >&2
  exit 1
fi

manifest_csv=$1
hdf5_dir=$2
output_pkl=$3
downsample_rate=${4:-3}
chunk_size=${5:-20}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_ROOT="${HY_VLA_ROOT:-${POLICY_DIR}/Hy-Embodied-0.5-VLA}"

if [[ ! -d "${HY_VLA_ROOT}" ]]; then
  echo "[hy_vla] Hy-Embodied source not found at ${HY_VLA_ROOT}. Run install.sh first." >&2
  exit 1
fi

cd "${HY_VLA_ROOT}"
echo "[hy_vla] computing norm stats -> ${output_pkl}"
uv run python scripts/compute_norm_hdf5.py \
  --csv "${manifest_csv}" \
  --hdf5-dir "${hdf5_dir}" \
  --output "${output_pkl}" \
  --downsample-rate "${downsample_rate}" \
  --chunk-size "${chunk_size}"
