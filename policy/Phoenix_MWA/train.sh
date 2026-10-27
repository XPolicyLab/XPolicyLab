#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# >= 6 )); then
  ACTION_DIM=$(bash "$P/../../utils/get_action_dim.sh" "$(cd "$P/../../.." && pwd)" "$3")
  if [[ "$ACTION_DIM" != 14 ]]; then
    echo "Phoenix-MWA RoboTwin training requires the registered 14D joint action profile" >&2
    exit 1
  fi
fi
export PYTHONPATH="$P/../../..:${PYTHONPATH:-}"
exec "${PYTHON:-python}" -m XPolicyLab.policy.Phoenix_MWA.training.train "$@"
