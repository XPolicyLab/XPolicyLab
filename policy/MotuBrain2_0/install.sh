#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${MOTUBRAIN_SOURCE_ROOT:-${script_dir}/source}"
if [[ ! -f "${source_dir}/pyproject.toml" ]]; then
    echo "Set MOTUBRAIN_SOURCE_ROOT to the standalone MotuBrain2.0-Robodojo-eval checkout." >&2
    exit 2
fi
# Install into the active dedicated policy environment. Never modifies the simulator env.
python -m pip install -e "${source_dir}"
