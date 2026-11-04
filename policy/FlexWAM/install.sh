#!/bin/bash
set -euo pipefail

# Install XPolicyLab plus the vendored FlexWAM package into the active
# environment. Install PyTorch first (see the policy README) — this script
# does not pin a CUDA wheel.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FLEXWAM_ROOT="${FLEXWAM_ROOT:-${SCRIPT_DIR}/flexwam}"

export PATH="/usr/bin:/usr/local/bin:${PATH}"

python -m pip install -e "${XPL_ROOT}"
python -m pip install -e "${FLEXWAM_ROOT}"
