#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
selector="${1:?Usage: runtime_python.sh auto|active|ENV_NAME|ENV_PATH|PYTHON [python args...]}"
shift
case "$selector" in
    auto) interpreter="$(crwm_python inference)" ;;
    active) interpreter="$(command -v python)" ;;
    *)
        if [[ -d "$selector" && -x "$selector/bin/python" ]]; then
            interpreter="$(cd "$selector" && pwd)/bin/python"
        elif [[ -x "$selector" && ! -d "$selector" ]]; then
            interpreter="$selector"
        else
            command -v conda >/dev/null || { echo "No interpreter/env at '$selector' and conda is unavailable." >&2; exit 2; }
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "$selector"
            interpreter="$(command -v python)"
        fi
        ;;
esac
exec "$interpreter" "$@"
