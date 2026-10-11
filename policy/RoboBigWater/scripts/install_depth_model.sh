#!/usr/bin/env bash
# Download the monocular depth model the policy server runs on the RGB images (Depth Anything V2, Metric-Indoor-Base,
# about 370 MB) into .cache/depth-model, unless ROBOSHELL_DEPTH in config.env already points at a model directory.
set -euo pipefail
source "$(dirname "$0")/sim_env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PIP_NO_INDEX
REPO=depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf
DEST=${ROBOSHELL_DEPTH:-$ROBOSHELL_ROOT/.cache/depth-model/$(basename "$REPO")}
if [ -f "$DEST/model.safetensors" ] && [ -f "$DEST/config.json" ]; then echo "depth model present: $DEST"; exit 0; fi
mkdir -p "$DEST"
PYTHONPATH="$ROBOSHELL_ROOT/.cache/policy-deps" "$ISAAC_PYTHON" - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
print("depth model downloaded to", sys.argv[2])
PY
