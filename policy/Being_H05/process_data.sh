#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

Links (or reuses) a LeRobot v2.1 dataset under policy/Being_H05/data/<4-tuple>/ and
registers it for Being-H training. The trailing expert_data_num is optional; when
omitted, all episodes are used. To ablate data scale, use a distinct ckpt_name
(e.g. myrun_50ep) and pass expert_data_num here.

Optional environment:
  LEROBOT_DATA_PATH   Source LeRobot repo (default: shared RoboDojo v21)
  RAW_DATA_ROOT       If set, print a hint to run XPolicyLab/scripts/transform_lerobot_v30_format.py first

Output layout (XPolicyLab convention):
  data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/
EOF
}

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}   # optional; empty = use all episodes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_TAG="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
DEST_DIR="${SCRIPT_DIR}/data/${DATA_TAG}"

SRC_DIR="${LEROBOT_DATA_PATH:?set LEROBOT_DATA_PATH to your RoboDojo LeRobot dataset dir}"

if [[ "${action_type}" != "joint" ]]; then
    echo -e "\033[31m[process_data] Being_H05 XPolicyLab flow currently supports action_type=joint only.\033[0m" >&2
    exit 1
fi

if [[ ! -d "${SRC_DIR}" ]]; then
    echo -e "\033[31m[process_data] LeRobot source not found: ${SRC_DIR}\033[0m" >&2
    if [[ -n "${RAW_DATA_ROOT:-}" ]]; then
        echo -e "\033[33m[process_data] Convert HDF5 with XPolicyLab/scripts/transform_lerobot_v30_format.py, then set LEROBOT_DATA_PATH.\033[0m"
    else
        echo -e "\033[33m[process_data] Set LEROBOT_DATA_PATH or convert raw data under ${ROOT_DIR}/data/${bench_name}/...\033[0m"
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
    ${expert_data_num:+--expert-data-num "${expert_data_num}"} \
    --action-type "${action_type}"

echo -e "\033[32m[process_data] ready: ${DATA_TAG}\033[0m"
