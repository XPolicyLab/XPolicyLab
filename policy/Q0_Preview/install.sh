#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt" -e "${XPL_ROOT}"
python - <<'PY'
import numpy, scipy, PIL, cv2, yaml, certifi, transforms3d
assert numpy.__version__ == '1.26.4'
print('Python dependencies installed. Continue with API, tool sandbox and Controller setup in policy/Q0_Preview/INSTALLATION.md.')
PY
