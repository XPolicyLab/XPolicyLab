#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGIBOT_DIR="${SCRIPT_DIR}/AgiBot-World"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "\033[33m[GO1 Install] Installing AgiBot-World (GO1) dependencies...\033[0m"

cd "${AGIBOT_DIR}"
pip install -e .

# flash-attn is optional here because wheel/build compatibility depends on the
# local torch/cuda toolchain. GO1 can fall back to eager attention at runtime.
if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    echo -e "\033[33m[GO1 Install] Installing flash-attn...\033[0m"
    if ! MAX_JOBS="${MAX_JOBS:-4}" pip install --no-build-isolation flash-attn; then
        echo -e "\033[33m[GO1 Install] flash-attn install failed, continuing without it.\033[0m"
    fi
else
    echo -e "\033[33m[GO1 Install] Skipping flash-attn. Set INSTALL_FLASH_ATTN=1 to try installing it.\033[0m"
fi

cd "${ROOT_DIR}"
pip install -e .

echo -e "\033[33m[GO1 Install] Installation complete.\033[0m"
