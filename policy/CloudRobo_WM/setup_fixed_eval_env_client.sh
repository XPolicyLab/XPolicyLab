#!/usr/bin/env bash
set -euo pipefail
if (($# < 10 || $# > 11)); then
    echo "Usage: $0 bench task ckpt env_cfg action seed device eval_env additional_info port [host]" >&2
    exit 2
fi
bench=$1; task=$2; ckpt=$3; embodiment=$4; action=$5; seed=$6
device=$7; eval_env=$8; info=$9; port=${10}; host=${11:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "$bench" == RoboTwin && "$embodiment" == arx_x5 && "$action" == joint ]] || exit 2
: "${CLOUDROBO_ROBOTWIN_ROOT:?Set the RoboTwin SDK root}"
: "${CLOUDROBO_FIXED_OUTPUT:?Set a new task output directory}"
export CUDA_VISIBLE_DEVICES="$device"
export PYTHONDONTWRITEBYTECODE=1
cache_root=${CLOUDROBO_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/cloudrobo_wm}
export CUDA_CACHE_PATH="${cache_root}/simulator_${port}/cuda"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/simulator_${port}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/simulator_${port}/triton"
mkdir -p "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
args=(--bench_name "$bench" --task_name "$task" --env_cfg_type "$embodiment"
 --policy_name "$(basename "$SCRIPT_DIR")" --host "$host" --port "$port"
 --protocol ws --eval_batch false --root_dir "$CLOUDROBO_ROBOTWIN_ROOT" --device_id "$device"
 --seed "$seed" --expert_check false --num_workers 1 --test_num 100
 --task_config "${CLOUDROBO_TASK_CONFIG:-demo_clean}"
 --instruction_type "${CLOUDROBO_INSTRUCTION_TYPE:-seen}")
if [[ -n "${ROBOTWIN_EVAL_ARGS_FILE:-}" ]]; then
    mapfile -t overrides < "$ROBOTWIN_EVAL_ARGS_FILE"
    args+=("${overrides[@]}")
fi
args+=(--additional_info "${info},ckpt_name=${ckpt},action_type=joint,num_workers=1,expert_check=false")
exec bash "$SCRIPT_DIR/runtime_python.sh" "$eval_env" "$SCRIPT_DIR/evaluation/eval_policy_xpolicylab.py" "${args[@]}"
