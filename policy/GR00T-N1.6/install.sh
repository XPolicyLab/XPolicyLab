#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GR00T_DIR="${SCRIPT_DIR}/Isaac-GR00T"

cd "${GR00T_DIR}"
pip install -e .

cd "${ROOT_DIR}/XPolicyLab"
pip install -e .

pip install PyYAML h5py pyarrow pandas opencv-python tqdm

echo "[GR00T-N1.6] Install step finished. If your machine needs CUDA/TensorRT-specific extras, follow Isaac-GR00T/README.md and scripts/deployment/*/install_deps.sh."
