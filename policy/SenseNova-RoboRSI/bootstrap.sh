#!/usr/bin/env bash
set -euo pipefail
unset CODEX_API_KEY
exec "${ROBORSI_BOOTSTRAP_PYTHON:-python3}" "$(dirname -- "${BASH_SOURCE[0]}")/bootstrap.py" "$@"
