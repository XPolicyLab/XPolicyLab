#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [norm_stats_args...]

Supported configuration: RoboDojo, arx_x5, joint.
Trailing arguments are forwarded to Spatial Forcing compute_norm_stats.py.
EOF
}

if [[ $# -lt 4 ]]; then
    usage
    exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
shift 4

if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[CSU-AI-0][ERROR] unsupported bench_name=${bench_name}; expected RoboDojo" >&2
    exit 2
fi
if [[ -z "${ckpt_name}" ]]; then
    echo "[CSU-AI-0][ERROR] ckpt_name must be non-empty" >&2
    exit 2
fi
if [[ "${env_cfg_type}" != "arx_x5" ]]; then
    echo "[CSU-AI-0][ERROR] unsupported env_cfg_type=${env_cfg_type}; expected arx_x5" >&2
    exit 2
fi
if [[ "${action_type}" != "joint" ]]; then
    echo "[CSU-AI-0][ERROR] unsupported action_type=${action_type}; expected joint" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SF_ROOT="${SCRIPT_DIR}/open_sf"
CONFIG_NAME="${CONFIG_NAME:-pi05sf_jax_robodojo_v21_offcache}"

if [[ ! -x "${OPEN_SF_ROOT}/.venv/bin/python" ]]; then
    echo "[CSU-AI-0][ERROR] Spatial Forcing environment is missing; run bash install.sh first" >&2
    exit 1
fi

echo "[CSU-AI-0] data: bench=${bench_name}, ckpt=${ckpt_name}, env=${env_cfg_type}, action=${action_type}"
echo "[CSU-AI-0] norm-stats config=${CONFIG_NAME}"

cd "${OPEN_SF_ROOT}"
exec .venv/bin/python scripts/compute_norm_stats.py "${CONFIG_NAME}" "$@"
