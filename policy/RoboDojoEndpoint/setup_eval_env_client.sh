#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 10 ]]; then
  echo 'Expected the ten standard environment-client arguments' >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
if [[ -f "$8/bin/activate" ]]; then
  source "$8/bin/activate"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$8"
fi
export PYTHONPATH="$BENCH_ROOT:$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
num_envs="${EVAL_NUM_ENVS:-8}"
[[ "$num_envs" =~ ^[1-8]$ ]] || { echo 'EVAL_NUM_ENVS must be between 1 and 8' >&2; exit 2; }
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
  exec python "$SCRIPT_DIR/debug_client.py" --env-cfg-type "$4" \
    --host "${11:-localhost}" --port "${10}" \
    --episodes "${DEBUG_EVAL_EPISODES:-2}" --num-envs "$num_envs"
fi
[[ "$1" == RoboDojo ]] || { echo 'This adapter supports RoboDojo simulation' >&2; exit 2; }
# Run the official evaluation entry point with an explicit environment count.
export CUDA_VISIBLE_DEVICES="$7"
cd "$BENCH_ROOT"
exec python -u src/eval_client/main.py --task_name "$2" --env_cfg_type "$4" \
  --num_envs "$num_envs" --policy_name RoboDojoEndpoint --host "${11:-localhost}" \
  --port "${10}" --protocol ws --additional_info "$9" --seed "$6" --device_id "$7" \
  --enable_cameras --kit_args '--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera' --headless
