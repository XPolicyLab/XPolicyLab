#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
xpl_root="$(cd "${script_dir}/../.." && pwd -P)"
: "${ACTWEAVE_ROOT:?Set ACTWEAVE_ROOT to the runtime checkout destination}"
if [[ ! -e "${ACTWEAVE_ROOT}" ]]; then
    git clone --branch main --single-branch https://github.com/ZJU-OmniAI/ActWeave.git "${ACTWEAVE_ROOT}"
fi
bash "${ACTWEAVE_ROOT}/scripts/install.sh"
python -m pip install -e "${xpl_root}"
python -m pip install httpx
