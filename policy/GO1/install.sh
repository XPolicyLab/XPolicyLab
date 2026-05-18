#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGIBOT_DIR="${SCRIPT_DIR}/AgiBot-World"

echo -e "\033[33m[GO1 Install] Installing AgiBot-World (GO1) dependencies...\033[0m"

# Install AgiBot-World package
cd "${AGIBOT_DIR}"
pip install -e .

# Install flash-attn (required for GO1 model)
echo -e "\033[33m[GO1 Install] Installing flash-attn...\033[0m"
MAX_JOBS=4 pip install --no-build-isolation flash-attn==2.4.2

# Install XPolicyLab package
cd "${SCRIPT_DIR}/../.."
pip install -e .

echo -e "\033[33m[GO1 Install] Installation complete.\033[0m"