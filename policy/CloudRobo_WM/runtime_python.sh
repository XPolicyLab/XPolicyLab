#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
selector=${1:?Usage: runtime_python.sh auto|active|ENV_NAME|ENV_PATH|PYTHON [arguments...]}
shift
case "$selector" in
    auto) interpreter=${CLOUDROBO_INFERENCE_PYTHON:-${SCRIPT_DIR}/source_inference/.venv/bin/python} ;;
    active) interpreter=$(command -v python) ;;
    *)
        if [[ -d "$selector" && -x "$selector/bin/python" ]]; then
            interpreter="$(cd "$selector" && pwd)/bin/python"
        elif [[ -x "$selector" && ! -d "$selector" ]]; then
            interpreter=$selector
        else
            command -v conda >/dev/null || { echo "The selected interpreter is unavailable." >&2; exit 2; }
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "$selector"
            interpreter=$(command -v python)
        fi
        ;;
esac
[[ -x "$interpreter" ]] || { echo "Run install.sh with a compatible framework interpreter." >&2; exit 2; }
exec "$interpreter" "$@"
