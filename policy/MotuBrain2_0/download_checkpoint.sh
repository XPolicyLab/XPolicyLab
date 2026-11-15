#!/usr/bin/env bash
set -euo pipefail
hf download Motubrain/MotuBrain2.0-Robodojo \
    step-5000.safetensors config.yaml action_stats.json \
    --revision 26c53046384721b21ef2b6d39b47b339cbc4bba8 \
    --local-dir "${1:?Usage: bash download_checkpoint.sh <destination>}"
