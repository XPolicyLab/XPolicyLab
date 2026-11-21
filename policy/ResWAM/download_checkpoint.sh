#!/bin/bash
set -euo pipefail

# Downloads the released RoboDojo checkpoint (stage-2 step 328k, the leaderboard / paper model)
# into policy/ResWAM/checkpoints/ResWAM-RoboDojo/, so that ckpt_name=ResWAM-RoboDojo resolves.
# Usage: bash download_checkpoint.sh [python]   (any python with huggingface_hub; default: python)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${1:-python}"
TARGET="${SCRIPT_DIR}/checkpoints/ResWAM-RoboDojo"

"${PYTHON_BIN}" - "${TARGET}" <<'PY'
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="OPTML-Group/ResWAM-RoboDojo",
    allow_patterns=["model.pt", "codec_statistics.json", "resolved_config.json", "README.md"],
    local_dir=sys.argv[1],
)
print(f"[download_checkpoint] {path}")
PY
