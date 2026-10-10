#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=${1:-vpp2}
INSTALL_MODE=${2:-eval}
if [[ "${INSTALL_MODE}" != eval && "${INSTALL_MODE}" != train ]]; then
  echo 'Usage: bash install.sh [env_name] [eval|train]' >&2
  exit 2
fi
TORCH_CUDA=${TORCH_CUDA:-cu130}
TORCH_VERSION=${TORCH_VERSION:-2.11.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.26.0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
conda install -c conda-forge 'ffmpeg>=6,<7' -y
python -m pip install --upgrade pip
python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
# Pin the public implementation used by this adapter.
VPP2_REV=ed398648fac3a0d7fb1e1eb9179db9d71cf326de
SOURCE_DIR="${SCRIPT_DIR}/upstream"
if [[ -e "${SOURCE_DIR}" ]]; then
  echo "${SOURCE_DIR} already exists; use its installed environment or move it before reinstalling." >&2
  exit 1
fi
git clone https://github.com/roboterax/video-prediction-policy-2.git "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" checkout --detach "${VPP2_REV}"
CONSTRAINTS_FILE=$(mktemp)
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
# Preserve the selected CUDA build while constraining the remaining packages.
sed -E '/^(torch|torchvision)==/d' "${SOURCE_DIR}/environment-reference.txt" > "${CONSTRAINTS_FILE}"
python -m pip install -e "${SOURCE_DIR}[eval]" -c "${CONSTRAINTS_FILE}"
if [[ "${INSTALL_MODE}" == train ]]; then
  python -m pip install -r "${SOURCE_DIR}/requirements-train.txt" -c "${CONSTRAINTS_FILE}"
fi
python -m pip install -e "${XPL_ROOT}" modelscope -c "${CONSTRAINTS_FILE}"
python -m pip check
python -c 'import torch, vpp2, XPolicyLab; print("VPP2 ready; torch", torch.__version__, "CUDA", torch.cuda.is_available())'
