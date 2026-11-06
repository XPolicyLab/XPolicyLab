#!/usr/bin/env bash
# Download a Griffin Alpha-S checkpoint's policy files from the Hugging Face Hub.
# Usage: bash download_checkpoint.sh [destination] [revision] [repo_id]
# Defaults: the RoboTwin 2.0 aloha-agilex release, griffinlabs/griffin-alpha-s-robotwin @ main,
# into checkpoints/griffin-alpha-s-robotwin. Pass the destination as ckpt_name to eval.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${1:-${SCRIPT_DIR}/checkpoints/griffin-alpha-s-robotwin}"
REVISION="${2:-main}"
REPO_ID="${3:-griffinlabs/griffin-alpha-s-robotwin}"
python - "${DESTINATION}" "${REVISION}" "${REPO_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=sys.argv[3], revision=sys.argv[2], local_dir=sys.argv[1],
    allow_patterns=["config.json", "generation_config.json", "model.safetensors", "policy_*.json", "policy_*.safetensors", "README.md", "LICENSE"],
)
print(f"[Griffin_Alpha_S] checkpoint downloaded to {path}")
PY
