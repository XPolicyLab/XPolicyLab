#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use the intended policy-server environment. Keep simulation/CUDA dependencies separate.
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
python -m pip install -e "${SCRIPT_DIR}/../.."
bash "${SCRIPT_DIR}/../WorldScape_Policy_2/install.sh"
