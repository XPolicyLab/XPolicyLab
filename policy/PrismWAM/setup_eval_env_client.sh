#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
if (($# < 10 || $# > 11)); then
    echo 'Usage: setup_eval_env_client.sh BENCH TASK CHECKPOINT ROBOT joint SEED GPU ENV INFO PORT [HOST]' >&2
    exit 2
fi
require_xpl
run_command bash "$XPL_ROOT/utils/setup_env_client.sh" "$XPL_ROOT/utils" \
    "$POLICY_DIR/deploy.yml" "$8" "${10}" "$1" "$2" "$4" PrismWAM "$9" \
    "$BENCH_ROOT" "$6" "$7" "${11:-localhost}"
