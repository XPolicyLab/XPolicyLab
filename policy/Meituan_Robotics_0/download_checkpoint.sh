#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ! "$2" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Usage: download_checkpoint.sh owner/repository COMMIT_SHA [destination]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON:-python3}" - "$1" "$2" "${3:-${SCRIPT_DIR}/checkpoints/mr0_width_130k}" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=str(Path(sys.argv[3]).expanduser().resolve()),
    allow_patterns=[
        'Meituan_Robotics_0/model.ckpt',
        'cdlam/model.pt',
        'base_vlm/Qwen3.5-4B-Action/*',
    ],
)
PY
