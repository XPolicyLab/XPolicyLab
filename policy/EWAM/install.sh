#!/bin/bash
set -euo pipefail

# Install XPolicyLab plus EWAM inference dependencies into the active
# environment. Install PyTorch first (see the policy README) — this script
# does not pin a CUDA wheel. The vendored EWAM/ tree is imported from source.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install -e "${XPL_ROOT}"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
