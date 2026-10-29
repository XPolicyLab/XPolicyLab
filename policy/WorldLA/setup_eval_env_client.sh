#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 10 && $# -le 11 ]] || { echo "Expected the 10 standard client args and optional host" >&2; exit 2; }
bench_name=$1 task_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
env_gpu_id=$7 eval_env=$8 additional_info=$9 port=${10} host=${11:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
worldla_activate_env "$eval_env"
export PYTHONPATH="$BENCH_ROOT:$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$env_gpu_id" PYTHONUNBUFFERED=1
source "$XPL_ROOT/utils/resolve_eval_env_type.sh"
mode="$(resolve_eval_env_type)"
if [[ "$mode" == debug ]]; then
    exec "$WORLDLA_PYTHON" "$XPL_ROOT/utils/debug_env_client.py" \
        --bench_name "$bench_name" --task_name "$task_name" --env_cfg_type "$env_cfg_type" \
        --policy_name WorldLA --host "$host" --port "$port" --protocol ws --eval_batch false
elif [[ "$mode" == sim && "$bench_name" == RoboTwin ]]; then
    # Native official evaluator; no asset, seed-validity, reward or physics edits.
    task_config=${WORLDLA_TASK_CONFIG:-demo_clean}
    case "$task_config" in
        demo_clean) instruction=seen ;;
        demo_randomized) instruction=unseen ;;
        *) echo "Unsupported task config: $task_config" >&2; exit 2 ;;
    esac
    exec "$WORLDLA_PYTHON" "$SCRIPT_DIR/run_robotwin.py" \
        "$BENCH_ROOT/scripts/eval_policy_xpolicylab.py" \
        --bench_name RoboTwin --task_name "$task_name" --policy_name WorldLA \
        --host "$host" --port "$port" --protocol ws --eval_batch false \
        --root_dir "$BENCH_ROOT" --device_id 0 --seed "$seed" \
        --task_config "$task_config" --test_num "${WORLDLA_TEST_NUM:-100}" \
        --instruction_type "$instruction" --additional_info "$additional_info"
else
    echo "This release is validated for RoboTwin simulation and XPolicyLab debug only." >&2
    exit 2
fi
