#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:---static}"
if (($#)); then shift; fi
case "$mode" in
  --static) exec python3 -B "$ROOT/tools/verify_submission.py" "$ROOT" ;;
  --debug)
    (($# == 4)) || { echo "Usage: $0 --debug checkpoint_root policy_gpu policy_env eval_env" >&2; exit 2; }
    for encoded in 0 1; do
      EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED="$encoded" bash "$ROOT/eval.sh" RoboTwin stack_bowls_two "$1" arx_x5 joint 0 "$2" "$2" "$3" "$4"
    done ;;
  --sim)
    (($# == 5)) || { echo "Usage: $0 --sim checkpoint_root policy_gpu sim_gpu policy_env eval_env" >&2; exit 2; }
    EVAL_ENV_TYPE=sim CLOUDROBO_TASK_CONFIG=demo_clean CLOUDROBO_INSTRUCTION_TYPE=seen CLOUDROBO_TEST_NUM=1 bash "$ROOT/eval.sh" RoboTwin adjust_bottle "$1" arx_x5 joint 0 "$2" "$3" "$4" "$5" ;;
  *) echo "Expected --static, --debug or --sim" >&2; exit 2 ;;
esac
