#!/usr/bin/env bash
set -euo pipefail

# Standard args, followed by converter options (see README).
bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
shift 4
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${POLICY_DIR}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
exec python -m XPolicyLab.policy.OpenDM.scripts.process_data \
    --bench-name "${bench_name}" --ckpt-name "${ckpt_name}" \
    --env-cfg-type "${env_cfg_type}" --action-type "${action_type}" "$@"
