#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$(dirname "$XPL_ROOT"):$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "${WSP_PYTHON:-python}" -B -u \
    -m XPolicyLab.policy.WorldScape_RSI.robodojo_server.online_service "$@"
