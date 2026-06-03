#!/bin/bash
set -euo pipefail

policy_conda_env=${1:-RISE}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${policy_conda_env}" python=3.11.14 -y || true
conda activate "${policy_conda_env}"

cd "${SCRIPT_DIR}/RISE"
bash install.sh

cd "${PROJECT_DIR}"
pip install -e .
