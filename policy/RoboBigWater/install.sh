#!/bin/bash
# RoboBigWater policy environment: the agent container image, the policy-server dependency (websockets) in the Isaac
# Sim Python, and the RGB-D observation configs. Settings come from config.env next to this file.
#   bash install.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/config.env" ]; then
  cp "${SCRIPT_DIR}/config.env.example" "${SCRIPT_DIR}/config.env"
  echo "[INSTALL] created config.env from config.env.example; fill in ISAAC_PYTHON, CODEX_BIN, MODEL_UPSTREAM, MODEL_BASE_PATH, MODEL_KEY_FILE and run install.sh again" >&2
  exit 1
fi
if grep -q "/path/to/" "${SCRIPT_DIR}/config.env"; then
  echo "[INSTALL] config.env still holds placeholder paths (/path/to/...); fill it in first" >&2
  exit 1
fi
exec "${SCRIPT_DIR}/roboshell.sh" setup
