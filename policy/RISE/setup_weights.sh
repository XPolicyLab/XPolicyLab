#!/bin/bash
set -euo pipefail

# Symlink shared Pi0.5 PyTorch weights into policy/RISE/weights/pi05_base_pytorch.
#
# Usage:
#   bash setup_weights.sh <source_pi05_pytorch_dir>
#
# Example:
#   bash setup_weights.sh /path/to/pi05_base_pytorch

usage="Usage: bash setup_weights.sh <source_pi05_pytorch_dir>"
source_dir=${1:?${usage}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_dir="${SCRIPT_DIR}/weights/pi05_base_pytorch"

if [[ ! -d "${source_dir}" ]]; then
    echo "[RISE] Source directory not found: ${source_dir}" >&2
    exit 1
fi

source_dir="$(cd "${source_dir}" && pwd)"
if [[ ! -f "${source_dir}/model.safetensors" && ! -f "${source_dir}/model.pt" ]]; then
    echo "[RISE] Expected model.safetensors or model.pt under: ${source_dir}" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/weights"
ln -sfn "${source_dir}" "${target_dir}"

echo "[RISE] Linked weights: ${target_dir} -> ${source_dir}"
