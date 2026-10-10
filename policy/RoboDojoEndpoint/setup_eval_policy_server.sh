#!/usr/bin/env bash
set -euo pipefail
# Standard policy-side arguments; the local process performs transport only.
if [[ $# -lt 9 ]]; then
  echo 'Expected the nine standard policy-server arguments' >&2
  exit 2
fi
[[ "$4" == arx_x5 && "$5" == joint ]] || {
  echo 'Supported configuration: arx_x5 / joint' >&2; exit 2;
}
: "${POLICY_ENDPOINT_URL:?Set POLICY_ENDPOINT_URL to the provided ws:// or wss:// endpoint}"
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
task_name="$2"
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
  task_name=None
fi
exec python "$SCRIPT_DIR/endpoint_server.py" --config_path "$SCRIPT_DIR/deploy.yml" \
  --overrides port="$9" host="${10:-127.0.0.1}" bench_name="$1" task_name="$task_name" \
  ckpt_name="$3" env_cfg_type="$4" action_type="$5" seed="$6" policy_name=RoboDojoEndpoint
