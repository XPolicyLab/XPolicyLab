#!/usr/bin/env bash
# Install the policy-server environment. One environment serves both the policy
# server and the WAM backend; see select_python.sh for how the interpreter is chosen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=select_python.sh
source "${SCRIPT_DIR}/select_python.sh"
_xpl_pick_python "$XPL_ROOT" || exit 1
[[ $XPL_CREATED == 1 ]] && echo "[info] created policy environment: $XPL_ENV_DIR"
echo "[info] policy environment: $XPL_PYTHON"
_xpl_check_python_version || exit 1

# ${PYTHON_BIN:+...} so wam/install.sh resolves the same interpreter it was told
# about here, instead of picking its own.
export PYTHON_BIN="$XPL_PYTHON"

"$XPL_PYTHON" -m pip install -e "${XPL_ROOT}"
"$XPL_PYTHON" -m pip install -r "${SCRIPT_DIR}/requirements-code.txt"

if [[ "${INSTALL_WAM:-1}" == "1" ]]; then
  if [[ -f "${SCRIPT_DIR}/wam/install.sh" ]]; then
    bash "${SCRIPT_DIR}/wam/install.sh" "${1:-install}"
  else
    echo "[warn] wam/install.sh not found; skipping the WAM inference environment" >&2
  fi
fi
