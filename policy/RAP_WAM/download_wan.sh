#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/runtime_env.sh"
exec "${RAP_WAM_PYTHON:-$RAP_WAM_ENV/bin/python}" "$DIR/download_wan.py" "$@"
