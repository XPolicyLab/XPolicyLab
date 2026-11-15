#!/usr/bin/env bash
set -euo pipefail
hf download Motubrain/MotuBrain2.0-Robodojo \
    step-5000.safetensors config.yaml action_stats.json \
    --revision 9939da9ccd7f9db9d62f1a56e2bcb9b3893cd0fe \
    --local-dir "${1:?Usage: bash download_checkpoint.sh <destination>}"
