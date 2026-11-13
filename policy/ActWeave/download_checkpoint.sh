#!/bin/bash
set -euo pipefail
: "${ACTWEAVE_ROOT:?Set ACTWEAVE_ROOT to the standalone runtime checkout}"
exec python "${ACTWEAVE_ROOT}/scripts/download_demonstrations.py" "$@"
