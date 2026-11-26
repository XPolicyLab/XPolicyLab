#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <bench> <ckpt> <env_cfg> <action> <seed> <gpu_ids> [training options; see --help]" >&2
    exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
sla_python uv
export CUDA_VISIBLE_DEVICES="$6"
action_dim="$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${BENCH_ROOT}" "$3")"
exec "${PYTHON_BIN}" -m XPolicyLab.policy.SLA.training train "$@" --action-dim "${action_dim}"
