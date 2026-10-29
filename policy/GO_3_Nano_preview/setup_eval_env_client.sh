#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"
usage() { echo 'Usage: setup_eval_env_client.sh <bench> <task> <ckpt> <profile> <action> <seed> <gpu> <eval_env> <additional_info> <port> [host]'; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# < 10 || $# > 11 )); then
  usage >&2
  exit 2
fi
source "$POLICY_DIR/configs/runtime.conf"
bench_name=${1}; task_name=${2}; ckpt_name=${3}; env_cfg_type=${4}; action_type=${5}
seed=${6}; env_gpu_id=${7}; eval_env=${8}; additional_info=${9}; port=${10}; host=${11:-localhost}
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
if [[ -x "${eval_env}/bin/python" ]]; then
  export PATH="${eval_env}/bin:$PATH"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$eval_env"
fi
source "$XPL_ROOT/utils/resolve_eval_env_type.sh"
mode="$(resolve_eval_env_type)"
if [[ "$mode" == sim && -n "${ROBOTWIN_ACTIVATE_SCRIPT:-}" ]]; then source "$ROBOTWIN_ACTIVATE_SCRIPT"; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
read -r eval_batch protocol < <(python - "$SCRIPT_DIR/deploy.yml" <<'PY'
import sys,yaml
d=yaml.safe_load(open(sys.argv[1]));print(str(d['eval_batch']).lower(),d['protocol'])
PY
)
if [[ "$mode" == debug ]]; then
  exec python "$XPL_ROOT/utils/debug_env_client.py" --bench_name "$bench_name" --task_name "$task_name" \
    --env_cfg_type "$env_cfg_type" --policy_name "$(basename "$SCRIPT_DIR")" --protocol "$protocol" \
    --host "$host" --port "$port" --eval_batch "$eval_batch" --eval_episode_num "${DEBUG_EPISODES:-2}"
elif [[ "$mode" == sim ]]; then
  exec bash "$BENCH_ROOT/scripts/eval_policy.sh" --bench_name "$bench_name" --task_name "$task_name" \
    --env_cfg_type "$env_cfg_type" --policy_name "$(basename "$SCRIPT_DIR")" --host "$host" --port "$port" \
    --protocol "$protocol" --eval_batch "$eval_batch" --root_dir "$BENCH_ROOT" \
    --device_id "$env_gpu_id" --additional_info "$additional_info" --seed "$seed"
else
  echo 'The released adapter supports debug and RoboTwin simulation.' >&2; exit 2
fi
