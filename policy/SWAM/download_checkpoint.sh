#!/usr/bin/env bash
set -euo pipefail
# Optional public download. Local evaluation should reuse existing weights.
target=${1:?Usage: download_checkpoint.sh OUTPUT_DIRECTORY}
python - "$target" <<'PY'
import sys
from huggingface_hub import snapshot_download
path=snapshot_download(repo_id="FreDERicia/SWAM",
    revision="ac5032420608a426176ed8de5568f7c79ad322c8",
    allow_patterns=["posttrain/*"],local_dir=sys.argv[1])
print(f"Checkpoint: {path}/posttrain")
print("Set SWAM_MODEL_PATH to this directory and set the component paths listed in README.md.")
PY
