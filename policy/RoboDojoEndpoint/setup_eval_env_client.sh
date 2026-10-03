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
export PYTHONPATH="$XPL_ROOT:$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
  exec python "$SCRIPT_DIR/debug_client.py" --env-cfg-type "$4" \
    --host "${11:-localhost}" --port "${10}" \
    --episodes "${DEBUG_EVAL_EPISODES:-2}"
fi
[[ "$1" == RoboDojo ]] || { echo 'This adapter supports RoboDojo simulation' >&2; exit 2; }
exec bash "$BENCH_ROOT/scripts/robodojo.sh" client --task "$2" \
  --policy-name RoboDojoEndpoint --policy-host "${11:-localhost}" --policy-port "${10}" \
  --ckpt "$3" --env-cfg "$4" --action-type "$5" --seed "$6" --env-gpu "$7"
