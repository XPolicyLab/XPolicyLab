#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${CSU_CHECKPOINT_ROOT:-${SCRIPT_DIR}/checkpoints/CSU-AI-0-v1}"
REPO_ID="${CSU_CHECKPOINT_REPO_ID:-}"
REVISION="${CSU_CHECKPOINT_REVISION:-main}"

if [[ -z "${REPO_ID}" ]]; then
    cat >&2 <<'EOF'
[CSU-AI-0][ERROR] Set CSU_CHECKPOINT_REPO_ID to the published Hugging Face
model/dataset repository containing the CSU-AI-0-v1 bundle.  The final public
repository ID must be filled in before the leaderboard PR is submitted.
EOF
    exit 2
fi

PYTHON_BIN="${CSU_DOWNLOAD_PYTHON:-$(command -v python3 || command -v python)}"
"${PYTHON_BIN}" -c 'import huggingface_hub' >/dev/null 2>&1 || {
    echo "[CSU-AI-0][ERROR] huggingface_hub is required" >&2
    exit 1
}
mkdir -p "${BUNDLE_DIR}"

REPO_ID="${REPO_ID}" REVISION="${REVISION}" BUNDLE_DIR="${BUNDLE_DIR}" \
"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["REPO_ID"],
    repo_type=os.environ.get("CSU_CHECKPOINT_REPO_TYPE", "model"),
    revision=os.environ["REVISION"],
    local_dir=os.environ["BUNDLE_DIR"],
)
PY

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_checkpoints.py" --root "${BUNDLE_DIR}"
