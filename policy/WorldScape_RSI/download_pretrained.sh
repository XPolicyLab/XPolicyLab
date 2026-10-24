#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 1 || "${1:-}" == "--help" ]]; then
    echo "Usage: bash download_pretrained.sh <external-model-directory>"
    [[ "${1:-}" == "--help" ]] && exit 0
    exit 1
fi
# Shared Stage-1 initialization used by WorldScape_Policy_2 and WorldScape_RSI.
exec hf download manifoldai-research/WorldScape-Policy-2 \
    --include 'wsp_2_pretrain/**' --local-dir "$1"
