#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> joint [expert_data_num] [options]" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/process_data.py" "$@"
