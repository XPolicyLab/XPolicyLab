#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
interpreter="${CLOUDROBO_DATA_PYTHON:-$ROOT/posttrain/.venv/bin/python}"
[[ -x "$interpreter" ]] || interpreter="${CLOUDROBO_DATA_PYTHON:-python3}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  exec "$interpreter" -B "$ROOT/posttrain/tools/prepare_data.py" --help
fi
if (($# < 4)); then
  echo 'Usage: bash process_data.sh RoboTwin <ckpt_name> arx_x5 joint --data-root PATH --bundle-root PATH [--index-only | --training-resources PATH --prepare-permissions]' >&2
  exit 2
fi
[[ "$1" == RoboTwin && -n "$2" && "$3" == arx_x5 && "$4" == joint ]] || { echo 'Expected RoboTwin <ckpt_name> arx_x5 joint' >&2; exit 2; }
shift 4
exec "$interpreter" -B "$ROOT/posttrain/tools/prepare_data.py" "$@"
