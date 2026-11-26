#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 1 || "${1:-}" == "--help" ]]; then
    echo "Usage: bash download_checkpoints.sh <external-model-directory>"
    [[ "${1:-}" == "--help" ]] && exit 0
    exit 1
fi
exec hf download manifoldai-research/WorldScape-R2SI \
    --local-dir "$1"
