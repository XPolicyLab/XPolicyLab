#!/usr/bin/env bash
# RDT_1B 一键安装（对应 INSTALLATION.md）
#
# 环境变量（可选）:
#   RDT_CONDA_ENV       conda 环境名，默认 rdt_1b
#   RDT_SKIP_CONDA_CREATE=1  跳过 conda create（环境已存在时）
#   RDT_SKIP_WEIGHTS=1       跳过权重准备（不下载、不软链）
#   RDT_WEIGHTS_SRC          已有权重根目录，软链到 weights/RDT/（优先于下载）

set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDT_ROOT="${POLICY_DIR}/rdt"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WEIGHTS_DIR="${POLICY_DIR}/weights/RDT"
RDT_CONDA_ENV="${RDT_CONDA_ENV:-rdt_1b}"

echo "[RDT_1B] RDT_ROOT=${RDT_ROOT}"
echo "[RDT_1B] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${RDT_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${RDT_CONDA_ENV}"; then
    echo "[RDT_1B] Creating conda env: ${RDT_CONDA_ENV}"
    conda create -n "${RDT_CONDA_ENV}" python=3.10 -y
  fi
fi

conda activate "${RDT_CONDA_ENV}"

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging==24.0 ninja
pip install flash-attn==2.7.2.post1 --no-build-isolation

cd "${RDT_ROOT}"
pip install -r requirements.txt

cd "${XPOLICYLAB_ROOT}"
pip install -e .

WEIGHT_NAMES=(t5-v1_1-xxl siglip-so400m-patch14-384 rdt-1b)

if [[ "${RDT_SKIP_WEIGHTS:-0}" != "1" ]]; then
  mkdir -p "${WEIGHTS_DIR}"
  if [[ -n "${RDT_WEIGHTS_SRC:-}" ]]; then
    for dir in "${WEIGHT_NAMES[@]}"; do
      src="${RDT_WEIGHTS_SRC}/${dir}"
      if [[ ! -e "${src}" ]]; then
        echo "[RDT_1B] Weight not found: ${src}" >&2
        exit 1
      fi
      ln -sfn "$(cd "${src}" && pwd)" "${WEIGHTS_DIR}/${dir}"
      echo "[RDT_1B] weights/RDT/${dir} -> $(readlink -f "${WEIGHTS_DIR}/${dir}")"
    done
  else
    if ! command -v huggingface-cli >/dev/null 2>&1; then
      pip install huggingface_hub
    fi
    cd "${WEIGHTS_DIR}"
    for repo in google/t5-v1_1-xxl google/siglip-so400m-patch14-384 robotics-diffusion-transformer/rdt-1b; do
      dir="$(basename "${repo}")"
      if [[ ! -e "${dir}" ]]; then
        echo "[RDT_1B] Downloading ${repo} -> ${WEIGHTS_DIR}/${dir}"
        huggingface-cli download "${repo}" --local-dir "${dir}"
      else
        echo "[RDT_1B] Skip existing ${dir}"
      fi
    done
  fi
fi

python -c "import XPolicyLab; print('XPolicyLab ok')" 2>/dev/null || true

echo "[RDT_1B] Installation finished."
echo "[RDT_1B] Activate: conda activate ${RDT_CONDA_ENV}"
echo "[RDT_1B] Weights dir: ${WEIGHTS_DIR}"
echo "[RDT_1B] Train: bash ${POLICY_DIR}/train.sh ..."
