#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# Output convention: data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
#
# Wall-X converts RoboDojo trajectories with its own converter (see the wall-x
# repository's tools/), which writes the vision / state / native EE action
# layout the checkpoint trains on. This script forwards to it when WALLX_REPO is
# set and is a stub otherwise (eval-only submission).
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WALLX_REPO="${WALLX_REPO:-${BENCH_ROOT}/wall-x}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

if [[ -d "${WALLX_REPO}" && -f "${WALLX_REPO}/tools/convert_robodojo.py" ]]; then
    exec python "${WALLX_REPO}/tools/convert_robodojo.py" \
        --bench "${bench_name}" --ckpt "${ckpt_name}" \
        --env-cfg "${env_cfg_type}" --action-type "${action_type}" \
        --output "${SCRIPT_DIR}/data/${data_setting}" \
        ${expert_data_num:+--limit "${expert_data_num}"}
fi

echo "[WallX] process_data stub: would write to ${SCRIPT_DIR}/data/${data_setting}"
echo "[WallX] Set WALLX_REPO to a wall-x checkout to run its converter."
