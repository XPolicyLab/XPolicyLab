#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HRDT_ROOT="${SCRIPT_DIR}/H_RDT"

conda create -n hrdt python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hrdt

cd "${HRDT_ROOT}"
pip install -r requirements.txt

cd "${ROOT_DIR}"
pip install -e .

cd "${HRDT_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download embodiedfoundation/H-RDT --local-dir ./
