#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -c "${SCRIPT_DIR}/constraints.txt" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c "${SCRIPT_DIR}/constraints.txt" -r "${SCRIPT_DIR}/requirements.txt"
python -m pip install -c "${SCRIPT_DIR}/constraints.txt" -e "${SCRIPT_DIR}/../.."
python -m pip check
echo "Next: python ${SCRIPT_DIR}/download_models.py"
