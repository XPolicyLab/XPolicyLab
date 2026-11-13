#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${CLOUDROBO_DATA_PYTHON:-python3}" -B "$SCRIPT_DIR/tools/download_initialization.py" "$@"
