#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ME_DEX_ROOT="${ME_DEX_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/me_dex_1_0/source}"
SOURCE_REPO="https://github.com/MachEmbodied/ME-Dex-1.0.git"

if [[ ! -d "${ME_DEX_ROOT}/.git" ]]; then
    git clone --depth 1 --branch main "${SOURCE_REPO}" "${ME_DEX_ROOT}"
fi

python -m pip install -e "${XPL_ROOT}"
python -m pip install -r "${ME_DEX_ROOT}/runtime/requirements.txt"
