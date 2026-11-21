#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${WSP_PYTHON:-python}"
exec "${python_bin}" "${SCRIPT_DIR}/tools/download_checkpoint.py" "$@"
