#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
#
# OpenWAM consumes native RoboDojo HDF5. This wrapper does not convert
# trajectories. Point OPENWAM_DATASET_DIR at an existing download, or launch
# the official interactive downloader.

bench_name=${1:?Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
expert_data_num=${5:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENWAM_ROOT="${OPENWAM_ROOT:-${POLICY_DIR}/OpenWAM}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_dir="${OPENWAM_DATASET_DIR:-${POLICY_DIR}/data/${data_setting}}"

if [[ "${action_type}" != "ee" ]]; then
    echo "[OpenWAM] ERROR: OpenWAM RoboDojo is an EE-space policy; action_type must be 'ee', got '${action_type}'." >&2
    exit 1
fi

echo "[OpenWAM] native RoboDojo HDF5 — no conversion."
echo "[OpenWAM] dataset_dir=${dataset_dir}"
if [[ "${env_cfg_type}" == "tianji_wuji" ]]; then
    echo "[OpenWAM] tianji_wuji: expects <dataset_dir>/<task>/tianji_wuji/data/episode_*.hdf5 with"
    echo "[OpenWAM]   state/{left,right}_ee_poses (T,7) and state/{left,right}_ee_joint_states (T,20)"
    echo "[OpenWAM]   (+ the same keys under action/ when recorded targets exist); trained as the 58-D WUJI hybrid."
fi
if [[ -n "${expert_data_num}" ]]; then
    echo "[OpenWAM] expert_data_num=${expert_data_num} is ignored (upstream-native data)."
fi

if [[ -d "${dataset_dir}" ]]; then
    echo "[OpenWAM] existing dataset dir found; nothing to download."
    exit 0
fi

DOWNLOADER="${OPENWAM_ROOT}/scripts/download_assets/download_benchmark_data.py"
if [[ ! -f "${DOWNLOADER}" ]]; then
    echo "[OpenWAM] ERROR: official downloader not found: ${DOWNLOADER}" >&2
    exit 1
fi

echo "[OpenWAM] launching official downloader (interactive)."
echo "[OpenWAM] select RoboDojo and store it at ${dataset_dir}, or set OPENWAM_DATASET_DIR afterwards."
cd "${OPENWAM_ROOT}"
python "${DOWNLOADER}"
