#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Output convention: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
#
# Eval-only submission: training lives in the wall-x repository and its release
# is tracked in the PR. This forwards to the upstream entrypoint when WALLX_REPO
# is set and otherwise only prepares the conventional output directory.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WALLX_REPO="${WALLX_REPO:-${BENCH_ROOT}/wall-x}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${SCRIPT_DIR}/checkpoints/${ckpt_setting}"

mkdir -p "${ckpt_dir}"

if [[ -d "${WALLX_REPO}" && -f "${WALLX_REPO}/tools/train.py" ]]; then
    exec env CUDA_VISIBLE_DEVICES="${gpu_id}" python "${WALLX_REPO}/tools/train.py" \
        --bench "${bench_name}" --ckpt "${ckpt_name}" \
        --env-cfg "${env_cfg_type}" --action-type "${action_type}" \
        --seed "${seed}" --output "${ckpt_dir}"
fi

echo "[WallX] train stub (eval-only submission): would write to ${ckpt_dir}"
echo "[WallX] Set WALLX_REPO to a wall-x checkout to run its training entrypoint."
