#!/bin/bash
# The one checkpoint PyRUA_Lean loads: SAM3 (segmentation), 3.4 GB, into ./checkpoints/sam3.pt,
# where setup_eval_policy_server.sh looks when SAM3_CHECKPOINT_PATH is not set.
#
#   bash download_assets.sh              # Hugging Face (facebook/sam3 is gated: accept its
#                                        # licence on the model page and `hf auth login` first)
#   bash download_assets.sh modelscope   # ModelScope mirror (facebook/sam3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/checkpoints"
mkdir -p "${TARGET}"

if [[ "${1:-huggingface}" == "modelscope" ]]; then
    python -m pip install -q modelscope
    modelscope download --model facebook/sam3 sam3.pt --local_dir "${TARGET}"
else
    python -m pip install -q "huggingface_hub>=0.34"
    hf download facebook/sam3 sam3.pt --local-dir "${TARGET}"
fi

test -s "${TARGET}/sam3.pt"
echo "[assets] ${TARGET}/sam3.pt"
