#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${1:-${SCRIPT_DIR}/checkpoints/Evo1_RoboTwin2_clean}"
REVISION="${2:-main}"
REPO_ID="${3:-MINT-SJTU/Evo1_RoboTwin2_clean}"
python - "${DESTINATION}" "${REVISION}" "${REPO_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=sys.argv[3],
    revision=sys.argv[2], local_dir=sys.argv[1],
    allow_patterns=["config.json", "norm_stats.json", "mp_rank_00_model_states.pt", "README.md"],
)
print(f"[Evo_1] Checkpoint downloaded to {path}")
PY
