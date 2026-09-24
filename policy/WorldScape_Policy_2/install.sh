#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# WorldScape Policy source lives in worldscape-policy/ next to this script.
# WORLDSCAPE_POLICY_ROOT can point at another checkout instead.
worldscape_root="${WORLDSCAPE_POLICY_ROOT:-${SCRIPT_DIR}/worldscape-policy}"
if [[ ! -d "${worldscape_root}/src/worldscape_policy" ]]; then
    echo "[INSTALL][ERROR] Invalid WorldScape checkout: ${worldscape_root}" >&2
    exit 2
fi
worldscape_root="$(cd "${worldscape_root}" && pwd)"

python -m pip install -e "${worldscape_root}[server]"
python -m pip install -e "${XPL_ROOT}"
