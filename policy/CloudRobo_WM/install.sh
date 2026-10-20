#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage="${1:-}"
if [[ "$stage" == --help || "$stage" == -h || -z "$stage" ]]; then
    echo "Usage: bash install.sh inference|posttrain --python /path/to/framework/python [--offline]"
    echo "Creates a separate source_<stage>/.venv inheriting your preinstalled Torch runtime."
    echo "Uses bundled source snapshots; --offline is retained for CLI compatibility. pip may still use its cache/network."
    exit 0
fi
[[ "$stage" == inference || "$stage" == posttrain ]] || { echo "Unknown stage: $stage" >&2; exit 2; }
shift
interpreter="python3"; offline=false
while (($#)); do
    case "$1" in
        --python) interpreter="${2:?Missing --python value}"; shift 2 ;;
        --offline) offline=true; shift ;;
        *) echo "Unknown install argument: $1" >&2; exit 2 ;;
    esac
done
arguments=("$stage" --python "$interpreter")
if [[ "$offline" == true ]]; then arguments+=(--offline); fi
"$interpreter" "$SCRIPT_DIR/install.py" "${arguments[@]}"
