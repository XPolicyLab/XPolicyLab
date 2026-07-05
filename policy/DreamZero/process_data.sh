#!/bin/bash
set -e
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

The trailing expert_data_num is optional; when omitted, all episodes are used.
To ablate data scale, use a distinct ckpt_name and pass expert_data_num here.

Optional environment overrides:
  DREAMZERO_DATA_DIR        Default: <policy>/data
  DREAMZERO_FPS             Default: 25
EOF
}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fps="${DREAMZERO_FPS:-25}"
output_dir="${DREAMZERO_DATA_DIR:-${SCRIPT_DIR}/data}"

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}   # optional; empty = use all episodes

python "${SCRIPT_DIR}/process_data.py" \
    --bench_name "${bench_name}" \
    --ckpt_name "${ckpt_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --action_type "${action_type}" \
    ${expert_data_num:+--expert_data_num "${expert_data_num}"} \
    --source_format hdf5 \
    --fps "${fps}" \
    --output_dir "${output_dir}"
