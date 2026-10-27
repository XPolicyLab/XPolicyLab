#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [--dry-run] [-- dotlist overrides...]" >&2
    exit 2
fi
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
export SIMATE_ROBOT_ACTION_DIM
SIMATE_ROBOT_ACTION_DIM="$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "$(dirname "${XPL_ROOT}")" "$3")"
exec "${PYTHON:-python}" "${POLICY_DIR}/training.py" train "$@"
