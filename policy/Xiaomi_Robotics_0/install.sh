# XPolicyLab deploy: policy server env=mibot; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR0_ROOT="${POLICY_DIR}/xiaomi_robotics_0/xr0"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${XR0_CONDA_ENV:-mibot}"

_resolve_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    return 0
  fi
  local conda_base
  conda_base="$(conda info --base)"
  if [[ -x "${conda_base}/bin/nvcc" ]]; then
    export CUDA_HOME="${conda_base}"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    return 0
  fi
  local candidate
  for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      return 0
    fi
  done
  return 1
}

_install_flash_attn() {
  local py_tag wheel_url
  py_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
  wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-${py_tag}-${py_tag}-linux_x86_64.whl"

  echo "[Xiaomi_Robotics_0] Trying prebuilt flash-attn wheel (${py_tag})..."
  if python -m pip install "${wheel_url}"; then
    return 0
  fi

  echo "[Xiaomi_Robotics_0] WARN: prebuilt wheel failed, trying source build..." >&2
  if _resolve_cuda_home; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "[Xiaomi_Robotics_0] CUDA_HOME=${CUDA_HOME}"
    python -m pip install flash-attn==2.8.3 --no-build-isolation && return 0
  else
    echo "[Xiaomi_Robotics_0] WARN: nvcc/CUDA_HOME not found, skip flash-attn (model falls back to sdpa)." >&2
  fi
  return 0
}

echo "[Xiaomi_Robotics_0] XR0_ROOT=${XR0_ROOT}"
echo "[Xiaomi_Robotics_0] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"
echo "[Xiaomi_Robotics_0] CONDA_ENV=${CONDA_ENV}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# conda deactivate hooks may reference unset vars; incompatible with `set -u`.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  conda create -n "${CONDA_ENV}" python=3.12 -y
fi
conda activate "${CONDA_ENV}"
# Some hosts leave base python on PATH after activate; force env bin first.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "[Xiaomi_Robotics_0] ERROR: expected Python 3.12 in ${CONDA_ENV}, got $(python --version)" >&2
  exit 1
fi
set -u

python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip uninstall -y ninja >/dev/null 2>&1 || true
python -m pip install ninja
_install_flash_attn

python -m pip install opencv-python-headless tqdm scipy

cd "${XR0_ROOT}"
python -m pip install -e .

cd "${XPOLICYLAB_ROOT}"
python -m pip install -e .
python -m pip install h5py pyyaml

echo "[Xiaomi_Robotics_0] Installation finished."
echo "[Xiaomi_Robotics_0] Training / eval / debug client all use: conda activate ${CONDA_ENV}"
