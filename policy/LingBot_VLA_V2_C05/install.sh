#!/bin/bash
set -euo pipefail

# Use a fresh Python 3.12 environment; do not modify a shared training environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install --no-deps -e "${XPL_ROOT}"
