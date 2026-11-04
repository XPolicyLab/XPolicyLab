#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
#
# FlexWAM does not consume native RoboDojo HDF5 or the official joint-space
# LeRobot export. Point FLEXWAM_DATASET_DIR at a LeRobot dataset whose keys are:
#   action.ee_pose / observation.ee_pose                         (20,)
#   observation.images.cam_high
#   observation.images.cam_left_wrist
#   observation.images.cam_right_wrist
# Each 20-D vector is left-then-right [xyz(3), rot6d first two rows(6), gripper(1)]
# in the per-arm robot base. Real arx_x5 / piper / piper_x poses are already in
# that frame. This script does not convert HDF5.

bench_name=${1:?Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
expert_data_num=${5:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_dir="${FLEXWAM_DATASET_DIR:-${POLICY_DIR}/data/${data_setting}}"

if [[ "${action_type}" != "ee" ]]; then
    echo "[FlexWAM] ERROR: FlexWAM RoboDojo is an EE-space policy; action_type must be 'ee', got '${action_type}'." >&2
    exit 1
fi

echo "[FlexWAM] custom LeRobot ee_pose layout — no HDF5 conversion in this script."
echo "[FlexWAM] dataset_dir=${dataset_dir}"
if [[ -n "${expert_data_num}" ]]; then
    echo "[FlexWAM] expert_data_num=${expert_data_num} is ignored."
fi

if [[ -d "${dataset_dir}" ]]; then
    echo "[FlexWAM] existing dataset dir found; nothing to download."
    exit 0
fi

echo "[FlexWAM] ERROR: LeRobot dataset not found at ${dataset_dir}." >&2
echo "[FlexWAM] Set FLEXWAM_DATASET_DIR to the prepared ee_pose dataset." >&2
exit 1
