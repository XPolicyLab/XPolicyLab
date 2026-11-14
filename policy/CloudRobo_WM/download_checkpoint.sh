#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${CLOUDROBO_DOWNLOAD_PYTHON:-python3}" "${SCRIPT_DIR}/download_checkpoint.py" "$@"
