#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$P/../../..:${PYTHONPATH:-}"
exec "${PYTHON:-python}" -m XPolicyLab.policy.Phoenix_MWA.training.data "$@"
