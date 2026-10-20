#!/usr/bin/env bash
set -euo pipefail
if (($# < 9 || $# > 10)); then
    echo "Usage: $0 bench task ckpt env_cfg action seed gpu policy_env port [host]" >&2; exit 2
fi
bench_name="$1"; task_name="$2"; ckpt_name="$3"; env_cfg_type="$4"; action_type="$5"
seed="$6"; policy_gpu_id="$7"; policy_env="$8"; port="$9"; host="${10:-localhost}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
export PYTHONPATH="$BENCH_ROOT:$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$policy_gpu_id"
exec bash "$SCRIPT_DIR/runtime_python.sh" "$policy_env" "$XPL_ROOT/setup_policy_server.py" \
    --config_path "$SCRIPT_DIR/deploy.yml" --overrides \
    "port=$port" "host=$host" "bench_name=$bench_name" "task_name=$task_name" \
    "ckpt_name=$ckpt_name" "env_cfg_type=$env_cfg_type" "seed=$seed" \
    "policy_name=CloudRobo_WM" "action_type=$action_type" "gpu_id=$policy_gpu_id"
