#!/usr/bin/env bash
set -euo pipefail
if (($# < 10 || $# > 11)); then
    echo "Usage: $0 bench task ckpt env_cfg action seed device eval_env additional_info port [host]" >&2
    exit 2
fi
bench_name=$1; task_name=$2; ckpt_name=$3; env_cfg_type=$4; action_type=$5
seed=$6; device_id=$7; eval_env=$8; additional_info=$9; port=${10}; host=${11:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "$SCRIPT_DIR")"
export CUDA_VISIBLE_DEVICES="$device_id"
export PYTHONDONTWRITEBYTECODE=1
cache_root=${CLOUDROBO_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/cloudrobo_wm}
cache_directory="${cache_root}/simulator_${port}"
export CUDA_CACHE_PATH="${cache_directory}/cuda"
export TORCHINDUCTOR_CACHE_DIR="${cache_directory}/torchinductor"
export TRITON_CACHE_DIR="${cache_directory}/triton"
mkdir -p "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [[ "$bench_name" != RoboTwin && "${EVAL_ENV_TYPE:-sim}" != debug ]]; then
    echo "This simulator adapter supports RoboTwin." >&2
    exit 2
fi
additional_info+=",num_workers=1,expert_check=true"
if [[ -z "${ROBOTWIN_EVAL_ARGS_FILE:-}" && -n "${CLOUDROBO_TASK_CONFIG:-}" ]]; then
    [[ "$CLOUDROBO_TASK_CONFIG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid task configuration." >&2; exit 2; }
    additional_info+=",task_config=${CLOUDROBO_TASK_CONFIG}"
fi
if [[ -z "${ROBOTWIN_EVAL_ARGS_FILE:-}" && -n "${CLOUDROBO_INSTRUCTION_TYPE:-}" ]]; then
    [[ "$CLOUDROBO_INSTRUCTION_TYPE" == seen || "$CLOUDROBO_INSTRUCTION_TYPE" == unseen ]] || { echo "Invalid instruction type." >&2; exit 2; }
    additional_info+=",instruction_type=${CLOUDROBO_INSTRUCTION_TYPE}"
fi
if [[ -z "${ROBOTWIN_EVAL_ARGS_FILE:-}" && -n "${CLOUDROBO_TEST_NUM:-}" ]]; then
    [[ "$CLOUDROBO_TEST_NUM" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid test count." >&2; exit 2; }
    additional_info+=",test_num=${CLOUDROBO_TEST_NUM}"
fi
bash "${XPL_ROOT}/utils/setup_env_client.sh" \
    "${XPL_ROOT}/utils" "${SCRIPT_DIR}/deploy.yml" "$eval_env" "$port" \
    "$bench_name" "$task_name" "$env_cfg_type" "$policy_name" "$additional_info" \
    "$BENCH_ROOT" "$seed" "$device_id" "$host"
