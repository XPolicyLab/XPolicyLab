#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mode=${1:-inference}
case "$mode" in
    inference) env_name=awomo05; requirements="$SCRIPT_DIR/requirements.txt" ;;
    training) env_name=awomo05-train; requirements="$SCRIPT_DIR/training/requirements.txt" ;;
    *) echo 'Usage: install.sh [inference|training]' >&2; exit 2 ;;
esac
source "$(conda info --base)/etc/profile.d/conda.sh"
env_list=$(conda env list)
if ! awk '{print $1}' <<< "$env_list" | grep -Fx "$env_name" >/dev/null; then
    conda create -n "$env_name" python=3.11 -y
fi
conda activate "$env_name"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "awomo05 requires Python 3.11"'
python -m pip install -e "$XPL_ROOT"
python -m pip install -r "$requirements"
echo "Installed policy environment: $env_name (Python 3.11)"
