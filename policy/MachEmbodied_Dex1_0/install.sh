#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MACH_EMBODIED_DEX_SOURCE_REPO="${MACH_EMBODIED_DEX_SOURCE_REPO:-https://github.com/Liuxuetao1219/MachEmbodied-Dex1.0.git}"
MACH_EMBODIED_DEX_SOURCE_REVISION="${MACH_EMBODIED_DEX_SOURCE_REVISION:-main}"
MACH_EMBODIED_DEX_CACHE_DIR="${MACH_EMBODIED_DEX_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/mach_embodied_dex1_0}"
MACH_EMBODIED_DEX_SOURCE_DIR="${MACH_EMBODIED_DEX_SOURCE_DIR:-${MACH_EMBODIED_DEX_CACHE_DIR}/source}"

mkdir -p "${MACH_EMBODIED_DEX_CACHE_DIR}"
if [[ ! -d "${MACH_EMBODIED_DEX_SOURCE_DIR}/.git" ]]; then
    git clone "${MACH_EMBODIED_DEX_SOURCE_REPO}" "${MACH_EMBODIED_DEX_SOURCE_DIR}"
fi
git -C "${MACH_EMBODIED_DEX_SOURCE_DIR}" fetch --depth 1 origin "${MACH_EMBODIED_DEX_SOURCE_REVISION}"
git -C "${MACH_EMBODIED_DEX_SOURCE_DIR}" checkout --detach FETCH_HEAD

python -m pip install -e "${XPL_ROOT}"
python -m pip install -r "${MACH_EMBODIED_DEX_SOURCE_DIR}/runtime/requirements.txt"
