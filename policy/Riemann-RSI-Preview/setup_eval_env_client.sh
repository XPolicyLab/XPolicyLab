#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 10 ]]; then
  printf '%s\n' 'Usage: setup_eval_env_client.sh BENCH TASK CKPT ROBOT ACTION SEED GPU ENV ADDITIONAL_INFO PORT [HOST]' >&2
  exit 2
fi
bench="$1"; task="$2"; checkpoint="$3"; robot="$4"; action="$5"; seed="$6"; gpu="$7"; environment="$8"; additional="$9"; port="${10}"; host="${11:-localhost}"
if [[ "$bench" != RoboDojo ]]; then
  printf '%s\n' 'This policy supports the RoboDojo benchmark' >&2
  exit 2
fi
if [[ -f "$environment/bin/python" ]]; then
  runner=("$environment/bin/python")
elif [[ -f "$environment" && -x "$environment" ]]; then
  runner=("$environment")
else
  runner=(conda run --no-capture-output -n "$environment" python)
fi
exec "${runner[@]}" -B "$ROOT/tools/run_official_eval.py" --physical-gpus "$gpu" --task_name "$task" --env_cfg_type "$robot" --policy_name Riemann-RSI-Preview --protocol ws --port "$port" --host "$host" --seed "$seed" --device_id 0 --headless --enable_cameras --additional_info "$additional"
