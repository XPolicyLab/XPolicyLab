#!/usr/bin/env bash
set -euo pipefail

# Usage: bash download_checkpoint.sh [destination_root]
# The checkpoint is saved under <destination_root>/60000/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="ZuoShun-AI/EpsilonVLA_RoboDoJo"
# Use the mainland Hugging Face mirror by default. Override if needed:
#   HF_ENDPOINT=https://huggingface.co bash download_checkpoint.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [destination_root]"
    echo "Default: ${SCRIPT_DIR}/checkpoints/EpsilonVLA_RoboDoJo/60000/"
    exit 0
fi
if (( $# > 1 )); then
    echo "Usage: $0 [destination_root]" >&2
    exit 1
fi

DEST_ROOT="${1:-${SCRIPT_DIR}/checkpoints/EpsilonVLA_RoboDoJo}"
if ! command -v hf >/dev/null 2>&1; then
    echo 'Missing hf CLI. Install with: python -m pip install -U huggingface_hub' >&2
    exit 1
fi

echo "Downloading ${REPO_ID}/60000 to ${DEST_ROOT}/60000"
# Re-running this command reuses files already downloaded successfully.
echo "Using Hugging Face endpoint: ${HF_ENDPOINT}"
hf download "$REPO_ID" \
    --repo-type model \
    --include '60000/**' \
    --local-dir "$DEST_ROOT"

if [[ ! -d "${DEST_ROOT}/60000/params" || ! -d "${DEST_ROOT}/60000/assets" ]]; then
    echo 'Checkpoint incomplete: expected 60000/params and 60000/assets in the repository.' >&2
    exit 1
fi
echo "Checkpoint downloaded: ${DEST_ROOT}/60000"
