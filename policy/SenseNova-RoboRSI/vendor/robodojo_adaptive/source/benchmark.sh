#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ROBORSI_BENCHMARK_ENV_SH:-}" ]]; then
    source "${ROBORSI_BENCHMARK_ENV_SH}"
fi
: "${ROBORSI_BENCHMARK_PYTHON:?Set ROBORSI_BENCHMARK_PYTHON to the prepared simulator interpreter}"
exec "${ROBORSI_BENCHMARK_PYTHON}" -u "${SCRIPT_DIR}/benchmark_bootstrap.py" "$@"
