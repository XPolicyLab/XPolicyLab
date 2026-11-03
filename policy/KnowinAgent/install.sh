#!/usr/bin/env bash
# Install KnowinAgent's Python dependencies into the RoboDojo client interpreter (the agent runs in the client process)
# and XPolicyLab's websocket transport into both interpreters. Codex and the perception services are installed
# separately (README).
#   bash install.sh /path/to/client/python [/path/to/policy-server/python]
set -euo pipefail

client_python=${1:?usage: bash install.sh CLIENT_PYTHON [SERVER_PYTHON]}
server_python=${2:-${client_python}}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PACKAGE_ROOT="$("${client_python}" "${SCRIPT_DIR}/_paths.py")"

pip_install() {
    local python=$1; shift
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${python}" "$@"
    else
        "${python}" -m pip install "$@"
    fi
}

# XPolicyLab's ws transport (the RoboDojo Isaac environment may ship an older websockets).
pip_install "${server_python}" "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" pyyaml
if [[ "${server_python}" != "${client_python}" ]]; then
    pip_install "${client_python}" "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8"
fi
# KnowinAgent itself, without touching the simulator's pinned numpy/torch.
pip_install "${client_python}" --no-deps -e "${PACKAGE_ROOT}"
"${client_python}" - <<'PY'
import importlib
for name in ("numpy", "PIL", "knowin_agent.backends.robodojo.official", "knowin_agent.bridge.host"):
    importlib.import_module(name)
print("[KnowinAgent][INSTALL] KnowinAgent imports in the client interpreter")
PY
echo "[KnowinAgent][INSTALL] next: install Codex CLI and log it in (KNOWIN_AGENT_CODEX_BIN, KNOWIN_AGENT_CODEX_HOME), and the perception"
echo "[KnowinAgent][INSTALL] services (KNOWIN_AGENT_SAM3_DIR, KNOWIN_AGENT_DEPTH_DIR); see ${SCRIPT_DIR}/README.md"
