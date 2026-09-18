#!/bin/bash
set -euo pipefail

export ME_DEX_ROOT="${ME_DEX_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/me_dex_1_0/source}"
runtime_path="${ME_DEX_ROOT}/runtime"
if [[ ! -d "${runtime_path}" ]]; then
    echo "[ERROR] ME-Dex-1.0 runtime not found: ${runtime_path}. Run install.sh first." >&2
    return 2 2>/dev/null || exit 2
fi
unset runtime_path
export PYTHONNOUSERSITE=1
export WAN_DISABLE_FLASH_ATTN="${WAN_DISABLE_FLASH_ATTN:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
