#!/usr/bin/env bash
set -euo pipefail
umask 000
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export NNODES="${NNODES:-18}"
export NPROC_PER_NODE=16
export CLOUDROBO_WM_NPU_DEVICES="${CLOUDROBO_WM_NPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export CLOUDROBO_WM_COMM_DTYPE="${CLOUDROBO_WM_COMM_DTYPE:-fp32}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$code_root/runtime/launch.py" "${1:-train}"
