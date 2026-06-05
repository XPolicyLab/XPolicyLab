# XPolicyLab deploy: policy server env=DM0; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEXBOTIC_ROOT="${POLICY_DIR}/dexbotic"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${DEXBOTIC_CONDA_ENV:-DM0}"

echo "[Dexbotic_DM0] DEXBOTIC_ROOT=${DEXBOTIC_ROOT}"
echo "[Dexbotic_DM0] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"
echo "[Dexbotic_DM0] CONDA_ENV=${CONDA_ENV}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
  conda activate "${CONDA_ENV}"
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
  pip install 'deepspeed>=0.18.0' 'numpydantic>=1.6'
fi

cd "${DEXBOTIC_ROOT}"
pip install -e .
pip install opencv-python-headless tqdm

cd "${XPOLICYLAB_ROOT}"
pip install -e .
pip install h5py pyyaml

python -c "import dexbotic; print('dexbotic ok')"
python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[Dexbotic_DM0] Installation finished."
echo "[Dexbotic_DM0] Next: hf download Dexmal/DM0-base --local-dir ${DEXBOTIC_ROOT}/checkpoints/DM0-base"
