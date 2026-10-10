#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"

python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install -r "${STARVLA_ROOT}/requirements.txt"
python -m pip install flash-attn --no-build-isolation
python -m pip install h5py pandas numpy==1.26.4 opencv-python==4.10.0.84
python -m pip install -e "${STARVLA_ROOT}"
