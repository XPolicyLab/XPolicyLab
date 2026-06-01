#!/bin/bash
set -euo pipefail

# Link an existing LeRobot v2.1 dataset (read-only) and compute RISE norm stats.
# Does not modify the source dataset.
#
# Usage:
#   bash process_lerobot.sh <source_lerobot_dir> [link_name]
#
# Example (RoboDojo):
#   bash process_lerobot.sh \
#     /mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_v21_video_abot \
#     RoboDojo_sim_v21_video_abot-lerobot

usage="Usage: bash process_lerobot.sh <source_lerobot_dir> [link_name]"
source_dir=${1:?${usage}}
link_name=${2:-$(basename "${source_dir}")-lerobot}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="${SCRIPT_DIR}/RISE/policy_and_value/policy_offline_and_value"
DATA_DIR="${SCRIPT_DIR}/data/${link_name}"
source_dir="$(cd "${source_dir}" && pwd)"

if [[ ! -d "${source_dir}/meta" || ! -d "${source_dir}/data" ]]; then
    echo "[RISE] Not a LeRobot dataset (missing meta/ or data/): ${source_dir}" >&2
    exit 1
fi

if [[ -e "${DATA_DIR}" && ! -L "${DATA_DIR}" ]]; then
    echo "[RISE] Refusing to overwrite non-symlink path: ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/data"
ln -sfn "${source_dir}" "${DATA_DIR}"

export RISE_LEROBOT_LAYOUT=robodojo
export RISE_XPOLICYLAB_DATASET="${DATA_DIR}"
export RISE_DEFAULT_PROMPT="${RISE_DEFAULT_PROMPT:-stack the bowls}"

echo "[RISE] Symlink: ${DATA_DIR} -> ${source_dir}"
echo "[RISE] Layout: ${RISE_LEROBOT_LAYOUT}"
echo "[RISE] Computing norm stats..."

cd "${OFFLINE_DIR}"
export PYTHONPATH="${OFFLINE_DIR}/src:${PYTHONPATH:-}"
python scripts/compute_norm_stats_fast.py --config-name Compute_norm

norm_stats="${OFFLINE_DIR}/data/norms/${link_name}/norm_stats.json"
echo "[RISE] Done. Dataset: ${DATA_DIR}"
echo "[RISE] Norm stats: ${norm_stats}"
