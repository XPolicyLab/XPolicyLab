#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || ${1:-} == "--help" ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <prepared_dataset_root> [prepared_dataset_root ...]"
    echo "Input: supervised LeRobot v2.1 or v3.0 datasets at 25 FPS."
    [[ ${1:-} == "--help" ]] && exit 0
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
shift 4
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${env_cfg_type} != "arx_x5" || ${action_type} != "joint" ]]; then
    echo "[DATA][ERROR] ME_Brain_1 supports arx_x5 joint datasets only." >&2
    exit 1
fi

exec "${FOCUS_VLWA_DATA_PYTHON:-python3}" "${POLICY_DIR}/process_data.py" \
    --dataset "$@" \
    --output-dir "${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
