#!/usr/bin/env bash
set -euo pipefail

# Training launcher for hy_vla.
#
# Hy-VLA training lives in the Hy-Embodied source tree (multi-node torchrun +
# Hydra config). This wrapper forwards to that recipe; tune the run via the
# documented env overrides (EXP_ID, EXP_ROOT, PRETRAIN, HDF5_DIR, NORM_PATH,
# NUM_MACHINES, NPROC_PER_NODE, CHIEF_IP, INDEX, ...).
#
# Single-node example:
#   CHIEF_IP=127.0.0.1 INDEX=0 NUM_MACHINES=1 NPROC_PER_NODE=8 \
#   HDF5_DIR=/path/to/robotwin/hdf5 EXP_ROOT=/path/to/experiments \
#   bash train.sh
#
# See the Hy-Embodied repo for full training docs:
#   https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_ROOT="${HY_VLA_ROOT:-${POLICY_DIR}/Hy-Embodied-0.5-VLA}"

if [[ ! -d "${HY_VLA_ROOT}" ]]; then
  echo "[hy_vla] Hy-Embodied source not found at ${HY_VLA_ROOT}. Run install.sh first." >&2
  exit 1
fi

cd "${HY_VLA_ROOT}"
echo "[hy_vla] launching Hy-Embodied RoboTwin/UMI training recipe"
exec bash scripts/train_robotwin_umi.sh "$@"
