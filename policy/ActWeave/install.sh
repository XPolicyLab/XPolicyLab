#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
xpl_root="$(cd "${script_dir}/../.." && pwd -P)"
: "${ACTWEAVE_ROOT:?Set ACTWEAVE_ROOT to the runtime checkout destination}"
: "${ACTWEAVE_REVISION:?Set ACTWEAVE_REVISION to the full ActWeave release commit ID}"
if [[ ! -e "${ACTWEAVE_ROOT}" ]]; then
    git clone --no-checkout https://github.com/ZJU-OmniAI/ActWeave.git "${ACTWEAVE_ROOT}"
    git -C "${ACTWEAVE_ROOT}" switch --detach "${ACTWEAVE_REVISION}"
fi
if [[ "$(git -C "${ACTWEAVE_ROOT}" rev-parse HEAD)" != "${ACTWEAVE_REVISION}" ]]; then
    printf '%s\n' 'ACTWEAVE_ROOT does not match ACTWEAVE_REVISION.' >&2
    exit 1
fi
bash "${ACTWEAVE_ROOT}/scripts/install.sh"
python -m pip install -e "${xpl_root}"
python -m pip install httpx
