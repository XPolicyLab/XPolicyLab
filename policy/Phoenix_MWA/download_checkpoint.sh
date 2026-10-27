#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$P/runtime/checkpoints.py" "${1:-$P/checkpoints/phoenix-robotwin-35k}" "${@:2}"
