#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR=${1:-"${SCRIPT_DIR}/checkpoints/Awomo06I"}
REPO=QZWang/Awomo-0.6-RoboTwin-AP338
REVISION=9b1ff0d9788767388e54e0deefd80ae6395cb505
mkdir -p "${TARGET_DIR}"
hf download "${REPO}" --revision "${REVISION}" --local-dir "${TARGET_DIR}"
python "${SCRIPT_DIR}/Awomo06I/scripts/verify_checkpoint.py" "${TARGET_DIR}"
echo "[Awomo06I] checkpoint verified: ${TARGET_DIR}"
