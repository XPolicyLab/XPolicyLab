#!/usr/bin/env bash
# Run the bundled Playground source with its own top-level Python packages.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
playground_python="${EMBODIEDEVO_PLAYGROUND_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$playground_python" ]]; then
    playground_python="$(command -v "$playground_python")"
fi
playground_python="$(cd "$(dirname "$playground_python")" && pwd)/$(basename "$playground_python")"
export PYTHONDONTWRITEBYTECODE=1
# The working directory and package path are local to the bundled source. Paths
# supplied by the caller (run directory, task definition, export) stay absolute.
exec "$playground_python" "$SCRIPT_DIR/playground/entrypoint.py" "$@"
