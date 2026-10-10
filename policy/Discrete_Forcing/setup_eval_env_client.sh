#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 10 || $# -gt 11 ]]; then
    echo "Usage: $0 bench task checkpoint env_cfg_type action_type seed env_gpu eval_env additional_info port [host]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec bash "$XPL_ROOT/utils/setup_env_client.sh" \
    "$XPL_ROOT/utils" "$SCRIPT_DIR/deploy.yml" "$8" "${10}" "$1" "$2" "$4" \
    Discrete_Forcing "$9" "${XPL_ROOT}/.." "$6" "$7" "${11:-localhost}"
