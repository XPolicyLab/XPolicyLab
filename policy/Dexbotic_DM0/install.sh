#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEXBOTIC_ROOT="${POLICY_DIR}/dexbotic"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[Dexbotic_DM0] DEXBOTIC_ROOT=${DEXBOTIC_ROOT}"
echo "[Dexbotic_DM0] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"

cd "${DEXBOTIC_ROOT}"
pip install -e .
pip install opencv-python-headless tqdm

cd "${XPOLICYLAB_ROOT}"
pip install -e .

python -c "import dexbotic; print('dexbotic ok')"
python -c "import XPolicyLab; print('XPolicyLab ok')"

echo "[Dexbotic_DM0] Installation finished."
echo "[Dexbotic_DM0] Next: hf download Dexmal/DM0-base --local-dir ${DEXBOTIC_ROOT}/checkpoints/DM0-base"
