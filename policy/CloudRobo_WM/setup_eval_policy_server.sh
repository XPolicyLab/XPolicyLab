#!/usr/bin/env bash
set -euo pipefail
if (($# < 9 || $# > 10)); then
    echo "Usage: $0 bench task ckpt env_cfg action seed device policy_env port [host]" >&2
    exit 2
fi
bench_name=$1; task_name=$2; ckpt_name=$3; env_cfg_type=$4; action_type=$5
seed=$6; device_id=$7; policy_env=$8; port=$9; host=${10:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XPL_ROOT="${CLOUDROBO_XPOLICYLAB_ROOT:-$XPL_ROOT}"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "$SCRIPT_DIR")"
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$device_id"
export PYTHONDONTWRITEBYTECODE=1
# Set the policy process environment before the interpreter and tensor runtime start.
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
cache_root=${CLOUDROBO_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/cloudrobo_wm}
cache_directory="${cache_root}/policy_${port}"
export CUDA_CACHE_PATH="${cache_directory}/cuda"
export TORCHINDUCTOR_CACHE_DIR="${cache_directory}/torchinductor"
export TRITON_CACHE_DIR="${cache_directory}/triton"
mkdir -p "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec bash "${SCRIPT_DIR}/runtime_python.sh" "$policy_env" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" --overrides \
    "port=$port" "host=$host" "bench_name=$bench_name" "task_name=$task_name" \
    "ckpt_name=$ckpt_name" "env_cfg_type=$env_cfg_type" "seed=$seed" \
    "policy_name=$policy_name" "action_type=$action_type" "gpu_id=$device_id"
