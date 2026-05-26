#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "[DreamZero install] Installing DreamZero package in editable mode."
cd "${SCRIPT_DIR}/dreamzero"
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

echo "[DreamZero install] Installing XPolicyLab package in editable mode."
cd "${ROOT_DIR}/XPolicyLab"
pip install -e .

echo "[DreamZero install] Done."
