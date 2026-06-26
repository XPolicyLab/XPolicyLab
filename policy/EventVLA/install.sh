#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENTVLA_ROOT="${SCRIPT_DIR}/source_eventvla"

if [[ "${SKIP_TORCH_INSTALL:-0}" != "1" ]]; then
    python -m pip install torch==2.6.0 torchvision==0.21.0
fi

python -m pip install -r "${EVENTVLA_ROOT}/requirements.txt"

if [[ "${SKIP_FLASH_ATTN_INSTALL:-0}" != "1" ]]; then
    python -m pip install flash-attn --no-build-isolation
fi

python -m pip install h5py pandas numpy==1.26.4 opencv-python==4.10.0.84
python -m pip install -e "${EVENTVLA_ROOT}"
