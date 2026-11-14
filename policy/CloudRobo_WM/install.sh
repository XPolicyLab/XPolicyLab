#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
interpreter=python3
arguments=()
while (($#)); do
    case "$1" in
        --python) interpreter=${2:?Missing interpreter}; shift 2 ;;
        --offline) arguments+=(--offline); shift ;;
        --help|-h) echo "Usage: bash install.sh --python /path/to/framework/python [--offline]"; exit 0 ;;
        *) echo "Unknown install argument: $1" >&2; exit 2 ;;
    esac
done
exec "$interpreter" "${SCRIPT_DIR}/install.py" "${arguments[@]}"
