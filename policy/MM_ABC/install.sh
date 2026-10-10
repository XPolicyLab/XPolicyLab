#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve_python.sh"
PY=${MMABC_PYTHON:-$(resolve_python "${1:-none}")}
MMABC_ROOT=${MMABC_ROOT:-"${SCRIPT_DIR}/MM_ABC"}

echo "[MM_ABC install] python=${PY}"
"${PY}" -m pip install "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" \
    "opencv-python-headless>=4.8" "pydantic>=2.5" pyyaml h5py
"${PY}" -m pip install -r "${MMABC_ROOT}/requirements.txt"
"${PY}" -m pip install einops safetensors   # VGGT-Omega, loaded from pretrained_ckpt/vggt-omega
"${PY}" -c "import torch, transformers, av, pyarrow, flash_attn; print('torch', torch.__version__, '| transformers', transformers.__version__, '| pyarrow', pyarrow.__version__, '| flash_attn', flash_attn.__version__)"
echo "[MM_ABC install] done"
