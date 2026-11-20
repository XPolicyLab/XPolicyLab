#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RIEMANN_INSTALL_PYTHON:-${RIEMANN_EVAL_PYTHON:-python3}}"
exec "$PYTHON" -B "$ROOT/tools/configure_official_policy.py" "$@"
