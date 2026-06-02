#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>

Links (or reuses) a LeRobot v2.1 dataset under policy/Being_H05/data/<5-tuple>/ and
registers it for Being-H training.

Optional environment:
  LEROBOT_DATA_PATH   Source LeRobot repo (default: shared RoboDojo v21)
  RAW_DATA_ROOT       If set, print a hint to run XPolicyLab/scripts/transform_lerobot_v30_format.py first

Output layout (XPolicyLab convention):
  data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>/
EOF
}

if [[ "$#" -ne 5 ]]; then
    usage >&2
    exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_TAG="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
DEST_DIR="${SCRIPT_DIR}/data/${DATA_TAG}"

DEFAULT_LEROBOT="/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21"
SRC_DIR="${LEROBOT_DATA_PATH:-${DEFAULT_LEROBOT}}"

if [[ "${action_type}" != "joint" ]]; then
    echo -e "\033[31m[process_data] Being_H05 XPolicyLab flow currently supports action_type=joint only.\033[0m" >&2
    exit 1
fi

if [[ ! -d "${SRC_DIR}" ]]; then
    echo -e "\033[31m[process_data] LeRobot source not found: ${SRC_DIR}\033[0m" >&2
    if [[ -n "${RAW_DATA_ROOT:-}" ]]; then
        echo -e "\033[33m[process_data] Convert HDF5 with XPolicyLab/scripts/transform_lerobot_v30_format.py, then set LEROBOT_DATA_PATH.\033[0m"
    else
        echo -e "\033[33m[process_data] Set LEROBOT_DATA_PATH or convert raw data under ${ROOT_DIR}/data/${dataset_name}/...\033[0m"
    fi
    exit 1
fi

if [[ ! -f "${SRC_DIR}/meta/episodes.jsonl" ]]; then
    echo -e "\033[31m[process_data] ${SRC_DIR} is not LeRobot v2.1 (missing meta/episodes.jsonl).\033[0m" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/data"
if [[ -e "${DEST_DIR}" && ! -L "${DEST_DIR}" ]]; then
    echo -e "\033[31m[process_data] ${DEST_DIR} exists and is not a symlink; remove it first.\033[0m" >&2
    exit 1
fi
ln -sfn "$(cd "${SRC_DIR}" && pwd)" "${DEST_DIR}"
echo -e "\033[33m[process_data] ${DEST_DIR} -> ${SRC_DIR}\033[0m"

python3 "${SCRIPT_DIR}/scripts/xpolicylab_dataset.py" prepare \
    --data-tag "${DATA_TAG}" \
    --data-path "${DEST_DIR}" \
    --expert-data-num "${expert_data_num}" \
    --action-type "${action_type}"

echo -e "\033[32m[process_data] ready: ${DATA_TAG}\033[0m"
