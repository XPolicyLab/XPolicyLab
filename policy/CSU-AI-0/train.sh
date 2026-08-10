#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [trainer_args...]

Supported configuration: RoboDojo, arx_x5, joint.
Trailing arguments are forwarded to Spatial Forcing train_align.py.
EOF
}

if [[ $# -lt 6 ]]; then
    usage
    exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

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
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[CSU-AI-0][ERROR] seed must be a non-negative integer: ${seed}" >&2
    exit 2
fi
if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "[CSU-AI-0][ERROR] gpu_id must be a non-negative integer: ${gpu_id}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SF_ROOT="${SCRIPT_DIR}/open_sf"

if [[ ! -x "${OPEN_SF_ROOT}/.venv/bin/python" ]]; then
    echo "[CSU-AI-0][ERROR] Spatial Forcing environment is missing; run bash install.sh first" >&2
    exit 1
fi

default_exp_name="csu_ai_0_${ckpt_name}_${env_cfg_type}_${action_type}_seed${seed}"
default_exp_name="${default_exp_name//\//_}"
default_exp_name="${default_exp_name// /_}"
CONFIG_NAME="${CONFIG_NAME:-pi05sf_jax_robodojo_v21_offcache}"
EXP_NAME="${EXP_NAME:-${default_exp_name}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${gpu_id}}"

export CONFIG_NAME EXP_NAME CUDA_VISIBLE_DEVICES

echo "[CSU-AI-0] train: bench=${bench_name}, ckpt=${ckpt_name}, env=${env_cfg_type}, action=${action_type}"
echo "[CSU-AI-0] seed=${seed}, gpu=${CUDA_VISIBLE_DEVICES}, config=${CONFIG_NAME}, exp=${EXP_NAME}"

cd "${OPEN_SF_ROOT}"
exec bash scripts/run_pi05sf_jax_offcache.sh --seed "${seed}" "$@"
