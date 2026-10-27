#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
if (($# < 9 || $# > 10)); then
    echo 'Usage: setup_eval_policy_server.sh BENCH TASK CHECKPOINT ROBOT joint SEED GPU POLICY_ENV PORT [HOST]' >&2
    exit 2
fi
require_xpl
bench="$1"; task="$2"; checkpoint="$3"; robot="$4"; action="$5"; seed="$6"
gpu="$7"; policy_env="$8"; port="$9"; host="${10:-localhost}"
[[ "$action" == joint ]] || { echo 'This adapter requires joint actions.' >&2; exit 2; }
overrides=("port=$port" "host=$host" "bench_name=$bench" "task_name=$task"
    "ckpt_name=$checkpoint" "env_cfg_type=$robot" "seed=$seed" "policy_name=PrismWAM"
    "action_type=$action" "gpu_id=$gpu")
[[ -z "${PRISMWAM_MODEL:-}" ]] || overrides+=("model_path=$PRISMWAM_MODEL")
[[ -z "${PRISMWAM_RECIPE:-}" ]] || overrides+=("recipe=$PRISMWAM_RECIPE")
[[ -z "${PRISMWAM_DEPTH_SCALE:-}" ]] || overrides+=("depth_scale=$PRISMWAM_DEPTH_SCALE")
[[ -z "${PRISMWAM_DEPTH_KEY:-}" ]] || overrides+=("depth_key=$PRISMWAM_DEPTH_KEY")
[[ -z "${PRISMWAM_DEVICE:-}" ]] || overrides+=("device=$PRISMWAM_DEVICE")
[[ -z "${PRISMWAM_DTYPE:-}" ]] || overrides+=("dtype=$PRISMWAM_DTYPE")
[[ -z "${PRISMWAM_INFERENCE_SEED:-}" ]] || overrides+=("inference_seed=$PRISMWAM_INFERENCE_SEED")
# Instructions are encoded with the T5 encoder under <weights>/Wan-AI/Wan2.2-TI2V-5B.
[[ -z "${PRISMWAM_ROOT:-}" ]] || export PRISMWAM_WEIGHTS_ROOT="${PRISMWAM_WEIGHTS_ROOT:-$PRISMWAM_ROOT/weights}"
if is_dry_run; then
    printf 'Policy environment: %s\n' "$policy_env"
else
    activate_policy_env "$policy_env"
fi
command=(env "CUDA_VISIBLE_DEVICES=$gpu" python "$XPL_ROOT/setup_policy_server.py"
    --config_path "$POLICY_DIR/deploy.yml" --overrides "${overrides[@]}")
show_command "${command[@]}"
if ! is_dry_run; then exec "${command[@]}"; fi
