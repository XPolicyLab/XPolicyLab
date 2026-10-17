#!/bin/bash
set -euo pipefail

# Install transport dependencies in the currently active Python environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install -e "${XPL_ROOT}" 'websockets>=16,<17'
