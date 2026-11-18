#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
worldscape_root="${SCRIPT_DIR}/worldscape-policy"
if [[ ! -d "${worldscape_root}/src/worldscape_policy" ]]; then
    echo "[INSTALL][ERROR] Missing bundled inference runtime: ${worldscape_root}" >&2
    exit 2
fi
python_bin="${WSP_PYTHON:-$(command -v python)}"
# Resolve both packages together so framework dependencies respect runtime pins.
"${python_bin}" -m pip install -e "${worldscape_root}[server]" -e "${XPL_ROOT}"
