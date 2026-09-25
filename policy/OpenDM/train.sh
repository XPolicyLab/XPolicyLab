#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
seed=${5:?seed required}
gpu_ids=${6:?comma-separated GPU ids required}
shift 6
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${POLICY_DIR}/../../.." && pwd)"
action_dim=$(bash "${WORKSPACE}/XPolicyLab/utils/get_action_dim.sh" "${WORKSPACE}" "${env_cfg_type}")
IFS=',' read -ra devices <<< "${gpu_ids}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export PYTHONPATH="${POLICY_DIR}/opendm:${WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${#devices[@]}" \
    --module XPolicyLab.policy.OpenDM.scripts.train \
    --bench-name "${bench_name}" --ckpt-name "${ckpt_name}" \
    --env-cfg-type "${env_cfg_type}" --action-type "${action_type}" \
    --seed "${seed}" --action-dim "${action_dim}" "$@"
