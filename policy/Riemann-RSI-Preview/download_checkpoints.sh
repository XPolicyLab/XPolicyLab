#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RIEMANN_EVAL_PYTHON:-python}"
exec "$PYTHON" "$ROOT/tools/download_assets.py" --manifest "$ROOT/configs/models/downloads.json" "$@"
