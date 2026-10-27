#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo 'Usage: train.sh [config.py] [launcher/trainer options]'
  echo '   or: train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [options]'
  echo 'GPU selection: CUDA_VISIBLE_DEVICES=0 or 0,1,...; otherwise detect all visible GPUs.'
  echo 'Options: --config PATH --run-name NAME --steps N --batch-size N --resume PATH --dry-run'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi

source "$POLICY_DIR/configs/runtime.conf"
source "$POLICY_DIR/tools/policy_env.sh"

args=()
if (( $# > 0 )) && [[ "$1" != -* && "$1" != *.py ]]; then
  if (( $# < 6 )); then usage >&2; exit 2; fi
  bench_name=$1; ckpt_name=$2; env_cfg_type=$3; action_type=$4; seed=$5; gpu_ids=$6
  shift 6
  [[ "${bench_name,,}" == robotwin && "$action_type" == joint ]] || { echo 'Supported: RoboTwin / joint' >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES="$gpu_ids"
  args=(--bench-name "$bench_name" --ckpt-name "$ckpt_name"
        --env-cfg-type "$env_cfg_type" --action-type "$action_type" --seed "$seed")
fi
go3_policy_environment
go3_require_policy_python
exec "$POLICY_PYTHON" -m go3_nano_preview.sft.launch "${args[@]}" "$@"
