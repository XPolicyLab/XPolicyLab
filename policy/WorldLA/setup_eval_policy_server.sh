#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 9 && $# -le 10 ]] || { echo "Expected the 9 standard server args and optional host" >&2; exit 2; }
bench_name=$1 task_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
policy_gpu_id=$7 policy_env=$8 policy_server_port=$9 policy_server_host=${10:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
worldla_activate_env "$policy_env"
overrides=("bench_name=$bench_name" "task_name=$task_name" "ckpt_name=$ckpt_name"
    "env_cfg_type=$env_cfg_type" "action_type=$action_type" "seed=$seed"
    "port=$policy_server_port" "host=$policy_server_host" "gpu_id=$policy_gpu_id")
if [[ -n "${WORLDLA_MODEL_DIR:-}" ]]; then overrides+=("model_dir=$WORLDLA_MODEL_DIR"); fi
exec env CUDA_VISIBLE_DEVICES="$policy_gpu_id" PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$WORLDLA_PYTHON" "$SCRIPT_DIR/serve.py" --config-path "$SCRIPT_DIR/deploy.yml" \
    --overrides "${overrides[@]}"
