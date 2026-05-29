#!/bin/bash
set -euo pipefail

policy_conda_env="${1:-LDA_1B}"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "${policy_conda_env}" python=3.10
conda activate "${policy_conda_env}"

cd "${POLICY_DIR}/LDA-1B"
pip install -r requirements.txt

MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install flash-attn --no-build-isolation --no-cache-dir

pip install -e .

# Make XPolicyLab importable in the policy env (required by model.py / process_data.py).
pip install -e "${PROJECT_ROOT}/XPolicyLab"
