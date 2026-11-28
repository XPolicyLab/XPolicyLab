#!/usr/bin/env bash
set -Eeuo pipefail
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == "system-deps" ]]; then
  [[ -f /etc/debian_version ]] || { echo 'system-deps requires Ubuntu/Debian'; exit 1; }
  installer_sudo=()
  if [[ "$(id -u)" != 0 ]]; then installer_sudo=(sudo); fi
  "${installer_sudo[@]}" apt-get update
  "${installer_sudo[@]}" apt-get install -y ca-certificates libgomp1
  exit 0
fi
if [[ "${1:-}" == check ]]; then
  target="${2:-$INSTALL_DIR/.venv}"
  exec "$target/bin/python" "$INSTALL_DIR/environment/install.py" "$@"
fi
exec "${INSTALL_PYTHON:-python3.11}" "$INSTALL_DIR/environment/install.py" "$@"
