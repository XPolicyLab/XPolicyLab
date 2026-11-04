#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XVLA_POLICY_DIR="${POLICY_DIR}/../X_VLA"

# GeoRefiner deliberately shares X-VLA's policy environment and imports its
# adapter. This keeps one copy of X-VLA and one model/checkpoint convention.
bash "${XVLA_POLICY_DIR}/install.sh"

echo "[GeoRefiner] Inference adapter installed in ${XVLA_CONDA_ENV:-XVLA}."
echo "[GeoRefiner] Run download_checkpoint.sh after the checkpoint is published."
