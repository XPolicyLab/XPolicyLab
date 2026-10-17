#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}"
WSP_INSTALLER="${XPL_ROOT}/policy/WorldScape_Policy_2/install.sh"
if [[ ! -f "${WSP_INSTALLER}" ]]; then
    echo "Install the adjacent RoboDojo-capable WorldScape_Policy_2 adapter first." >&2
    exit 1
fi
bash "${WSP_INSTALLER}"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
echo "Prepare the matching RoboDojo/CuRobo runtime and bind both WSP checkpoints; see README.md."
