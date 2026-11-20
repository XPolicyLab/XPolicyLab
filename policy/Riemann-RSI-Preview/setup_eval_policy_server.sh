#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 9 ]]; then
  printf '%s\n' 'Usage: setup_eval_policy_server.sh BENCH TASK CKPT ROBOT ACTION SEED GPU POLICY_ENV PORT [HOST]' >&2
  exit 2
fi
bench="$1"; task="$2"; checkpoint="$3"; robot="$4"; action="$5"; seed="$6"; gpu="$7"; environment="$8"; port="$9"; host="${10:-localhost}"
if [[ -f "$environment/bin/python" ]]; then
  runner=("$environment/bin/python")
elif [[ -f "$environment" && -x "$environment" ]]; then
  runner=("$environment")
else
  runner=(conda run --no-capture-output -n "$environment" python)
fi
exec "${runner[@]}" -B "$ROOT/tools/serve_official_policy.py" --physical-gpus "$gpu" --mode "$task" --port "$port" --host "$host" --config "$ROOT/deploy.yml" --checkpoint "$checkpoint" --bench_name "$bench" --env_cfg_type "$robot" --action_type "$action" --seed "$seed"
