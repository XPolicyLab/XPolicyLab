#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Use a dedicated Python 3.11 environment for the policy")
PY
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
if [[ -f "${XPL_ROOT}/setup_policy_server.py" ]]; then
    python -m pip install -e "${XPL_ROOT}"
fi
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
echo "Installed. Run python download_checkpoints.py to retrieve the pinned backbone and tokenizers."
