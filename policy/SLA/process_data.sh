#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <bench> <ckpt> <env_cfg> <action> <lerobot_dir_or_glob> [--max-frames N] [--batch-size N] [--num-workers N]" >&2
    exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
sla_python uv
exec "${PYTHON_BIN}" -m XPolicyLab.policy.SLA.training prepare "$@"
