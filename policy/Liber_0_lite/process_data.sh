#!/usr/bin/env bash
set -euo pipefail
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$POLICY_DIR/../.." && pwd)"
export PYTHONPATH="$(dirname "$XPL_ROOT"):$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m XPolicyLab.policy.Liber_0_lite.liber0.training.entrypoint \
    --policy-dir "$POLICY_DIR" data "$@"
