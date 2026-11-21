#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${1:-${SCRIPT_DIR}/.venv}"
"${PYTHON_BIN:-python3}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
"${ENV_DIR}/bin/python" -m pip install -c "${SCRIPT_DIR}/requirements-transport.txt" -e "${SCRIPT_DIR}/runtime" -e "${XPL_ROOT}"
echo "Policy environment: ${ENV_DIR}"
