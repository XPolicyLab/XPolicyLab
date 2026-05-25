#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A1_DIR="${SCRIPT_DIR}/A1"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "\033[33m[A1 Install] Installing A1 package...\033[0m"
cd "${A1_DIR}"
pip install -e .[all]

# Keep a separate requirements sync as opt-in to avoid silently overriding
# versions already resolved from pyproject extras.
if [ "${INSTALL_A1_REQUIREMENTS:-0}" = "1" ]; then
    pip install -r requirements.txt
else
    echo -e "\033[33m[A1 Install] Skipping requirements.txt. Set INSTALL_A1_REQUIREMENTS=1 to install pinned extras.\033[0m"
fi

echo -e "\033[33m[A1 Install] Installing XPolicyLab package...\033[0m"
cd "${ROOT_DIR}"
pip install -e .

echo -e "\033[33m[A1 Install] Installation complete.\033[0m"
