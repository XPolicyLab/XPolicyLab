#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${CLOUDROBO_DATA_PYTHON:-python3}" -B "$ROOT/tools/download_data.py" "$@"
