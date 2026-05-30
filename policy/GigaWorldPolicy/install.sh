#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIGA_ROOT="${POLICY_DIR}/giga_world_policy"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${GIGAWORLD_CONDA_ENV:-gigaworld-policy}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${GIGAWORLD_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.11 -y
  fi
fi

conda activate "${CONDA_ENV}"

cd "${GIGA_ROOT}"
pip install -e ./third_party/giga-train
pip install -e ./third_party/giga-models
pip install -e ./third_party/giga-datasets
pip install -r ./third_party/giga-train/requirements.txt 2>/dev/null || true
pip install -r ./third_party/giga-models/requirements.txt 2>/dev/null || true
pip install -r ./third_party/giga-datasets/requirements.txt 2>/dev/null || true

cd "${XPOLICYLAB_ROOT}"
pip install -e .

python -c "import XPolicyLab; print('XPolicyLab ok')"
python -c "import giga_train; print('giga_train ok')" 2>/dev/null || true

echo "[GigaWorldPolicy] Done. conda activate ${CONDA_ENV}"
