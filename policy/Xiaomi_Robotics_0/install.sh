# XPolicyLab deploy: policy server env=mibot; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR0_ROOT="${POLICY_DIR}/xiaomi_robotics_0/xr0"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${XR0_CONDA_ENV:-mibot}"

echo "[Xiaomi_Robotics_0] XR0_ROOT=${XR0_ROOT}"
echo "[Xiaomi_Robotics_0] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"
echo "[Xiaomi_Robotics_0] CONDA_ENV=${CONDA_ENV}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  conda create -n "${CONDA_ENV}" python=3.12 -y
fi
conda activate "${CONDA_ENV}"

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip uninstall -y ninja >/dev/null 2>&1 || true
pip install ninja
pip install flash-attn==2.8.3 --no-build-isolation || true

pip install opencv-python-headless tqdm scipy

cd "${XR0_ROOT}"
pip install -e .

cd "${XPOLICYLAB_ROOT}"
pip install -e .
pip install h5py pyyaml

echo "[Xiaomi_Robotics_0] Installation finished."
echo "[Xiaomi_Robotics_0] Training / eval / debug client all use: conda activate ${CONDA_ENV}"
