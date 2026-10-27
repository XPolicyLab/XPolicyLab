#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
if (($# < 5)); then
    echo 'Usage: process_data.sh BENCH CHECKPOINT ROBOT joint PREPARE_SUBCOMMAND [ARGS...]' >&2
    exit 2
fi
require_prismwam
[[ "$4" == joint ]] || { echo 'Use the native PrismWAM tools for non-joint recipes.' >&2; exit 2; }
shift 4
run_command "${PYTHON:-python}" "$PRISMWAM_ROOT/scripts/prepare_data.py" "$@"
