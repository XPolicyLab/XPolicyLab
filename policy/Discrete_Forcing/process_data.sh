#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "Usage: DF_DATA_ROOT=/prepared/robotwin bash $0 RoboTwin clean env_cfg_type joint" >&2
    exit 2
fi
[[ "$1" == RoboTwin && "$4" == joint ]] || { echo "Only RoboTwin joint data are supported." >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_root="${DF_ROOT:-${SCRIPT_DIR}/source_discrete_forcing}"
python "$SCRIPT_DIR/prepare_data.py" "$model_root" "${DF_DATA_ROOT:?Set DF_DATA_ROOT to prepared RoboTwin clean data}" \
    "$SCRIPT_DIR/data/$1-$2-$3-$4"
