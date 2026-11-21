#!/usr/bin/env bash
# Official RoboDojo smoke run with both the simulator client and the policy
# server in the project environment. Run it after any dependency change.
#
#   bash environment/smoke.sh [GPU] [TASK]    # defaults: GPU 0, stack_bowls
#
# Uses XPolicyLab's demo_policy, one episode, eval seed 0. It checks that the
# stack runs end to end; the score is not a benchmark result.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:-0}"
TASK="${2:-stack_bowls}"
# shellcheck disable=SC1091
source "$ROOT/environment/activate.sh"
[ -f "$HARNESS_ROBODOJO_ROOT/scripts/robodojo.sh" ] || { echo "RoboDojo not found at $HARNESS_ROBODOJO_ROOT" >&2; exit 1; }

OUT="${TMPDIR:-/tmp}/harness-env-smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
cd "$HARNESS_ROBODOJO_ROOT"
bash scripts/robodojo.sh smoke \
  --policy-dir "$HARNESS_ROBODOJO_ROOT/XPolicyLab/policy/demo_policy" \
  --ckpt demo \
  --policy-env "${HARNESS_ENV:-harness-policy}" \
  --eval-env "${HARNESS_ENV:-harness-policy}" \
  --only "$TASK" \
  --eval-num 1 \
  --seed 0 \
  --env-cfg arx_x5 \
  --action-type joint \
  --policy-gpu "$GPU" \
  --env-gpu "$GPU" \
  --run-id "harness-env-smoke-$TASK" \
  --summary "$OUT/summary.json" \
  --markdown "$OUT/summary.md"
echo "Smoke summary: $OUT/summary.md"
cat "$OUT/summary.md"
