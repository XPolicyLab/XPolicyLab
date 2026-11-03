#!/bin/bash
set -euo pipefail

# Convert RoboDojo demonstrations into the LeRobot repo flex-pi trains on.
#
# This delegates to the OFFICIAL XPolicyLab converter
# (scripts/transform_lerobot_v21_format.py) rather than shipping its own: the
# dataset keys flex-pi wants are exactly the ones that converter emits —
# observation.state, action, observation.images.cam_high / cam_left_wrist /
# cam_right_wrist. No key remapping is needed anywhere.
#
# NOTE: the official converter reads only `vision/<cam>/colors`, so the LeRobot
# export it produces carries NO depth. That is fine for the RGB-only preset
# (configs/task/robodojo_unified_flex_3cam_384_1e-4.yaml). Training the
# pointmap stream needs the separate depth export (data/RoboDojo_depth on HF)
# and a converter that carries `vision/<cam>/depths` — see README.
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
#                        [max_episode] [raw_task_pattern]
#
# raw_task_pattern defaults to "*" (every task under data/<bench_name>/).
# Source data is expected at <BENCH_ROOT>/data/<bench_name>/<task>/<env_cfg_type>/data.

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
max_episode=${5:-200}
raw_task_pattern=${6:-"*"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

dataset_id="${FLEXPI_DATASET_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
link_root="${SCRIPT_DIR}/data/${dataset_id}"

# flex-pi trains at 240x320 (configs/data/robodojo.yaml raw_shape), and the
# adapter resizes deploy frames to the same size, so pin it here instead of
# letting the converter auto-detect from the first source frame.
resolution="${FLEXPI_CONVERT_RESOLUTION:-240x320}"

echo "[process_data] source : ${BENCH_ROOT}/data/${bench_name}/${raw_task_pattern}/${env_cfg_type}"
echo "[process_data] repo_id: ${dataset_id}"
echo "[process_data] resize : ${resolution}"

python "${XPL_ROOT}/scripts/transform_lerobot_v21_format.py" \
    "${bench_name}.${raw_task_pattern}.${env_cfg_type}" \
    --repo_id "${dataset_id}" \
    --data_type "${bench_name}" \
    --max_episode "${max_episode}" \
    --resolution "${resolution}"

# The converter writes into the LeRobot cache (HF_LEROBOT_HOME/<repo_id>).
# Link it to a stable path so train.sh does not have to know that layout.
lerobot_dir="$(python - "${dataset_id}" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
print(HF_LEROBOT_HOME / sys.argv[1])
PY
)"

if [[ ! -d "${lerobot_dir}/meta" ]]; then
    echo "[ERROR] conversion produced no dataset at ${lerobot_dir}" >&2
    exit 1
fi

mkdir -p "${link_root}"
ln -sfn "${lerobot_dir}" "${link_root}/lerobot"

echo
echo "[OK] LeRobot dataset: ${link_root}/lerobot -> ${lerobot_dir}"
echo "dataset_stats.json is written by flex-pi's RobotVideoDataset on the first"
echo "training run, so there is no separate stats step."
