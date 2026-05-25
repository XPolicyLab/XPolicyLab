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
pip install -e .

# Make XPolicyLab importable in the policy env (required by model.py / process_data.py).
pip install -e "${PROJECT_ROOT}/XPolicyLab"

# Re-pin core deps before flash-attn install so transitive resolutions don't drift
# torch / transformers to versions that break LDA-1B's dataclass(PretrainedConfig).
pip install 'torch==2.6.0' 'torchvision==0.21.0' 'transformers==4.57.1' 'huggingface-hub==0.36.0'

# Install a prebuilt flash-attn wheel matching torch 2.6 + cu12 + cxx11abi=FALSE.
# Dao-AILab's flash-attn >=2.8 wheels require the NEW std::__cxx11 ABI even in
# their cxx11abiFALSE variant, which is incompatible with PyPI's torch 2.6
# (OLD ABI -> c10::Error takes std::basic_string, not std::__cxx11::basic_string).
# 2.7.4.post1 is the highest published version that links cleanly. --no-deps
# prevents pip from silently bumping torch to 2.12.
pip install --no-deps \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
