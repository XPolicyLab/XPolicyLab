#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_ROOT="${POLICY_DIR}/lingbot_vla"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${LINGBOT_VLA_CONDA_ENV:-lingbot_vla}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${LINGBOT_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.12 -y
  fi
fi

conda activate "${CONDA_ENV}"

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# LingBot-VLA requires lerobot at a fixed git commit (provides lerobot.common).
LEROBOT_GIT_DIR="${LINGBOT_ROOT}/.vendor/lerobot"
LEROBOT_GIT_COMMIT="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
if [[ ! -d "${LEROBOT_GIT_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${LEROBOT_GIT_DIR}")"
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git "${LEROBOT_GIT_DIR}"
fi
git -C "${LEROBOT_GIT_DIR}" fetch --depth 1 origin "${LEROBOT_GIT_COMMIT}" 2>/dev/null || true
git -C "${LEROBOT_GIT_DIR}" checkout "${LEROBOT_GIT_COMMIT}"
pip install -e "${LEROBOT_GIT_DIR}"

cd "${LINGBOT_ROOT}"
git submodule update --init --recursive 2>/dev/null || true

# flash-attn wheel: set FLASH_ATTN_WHEEL_URL or place wheel in policy dir
if [[ -n "${FLASH_ATTN_WHEEL_URL:-}" ]]; then
  wget -q -O /tmp/flash_attn.whl "${FLASH_ATTN_WHEEL_URL}"
  pip install /tmp/flash_attn.whl
elif ls "${POLICY_DIR}"/flash_attn*.whl 1>/dev/null 2>&1; then
  pip install "${POLICY_DIR}"/flash_attn*.whl
else
  echo "[LingBot_VLA] Warning: flash-attn wheel not found; pip install flash-attn manually if needed"
fi

pip install -e .
pip install -r requirements.txt
cd lingbotvla/models/vla/vision_models/lingbot-depth/
pip install -e . --no-deps
cd ../MoGe
pip install -e .
cd "${LINGBOT_ROOT}"

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[LingBot_VLA] Done. conda activate ${CONDA_ENV}"
