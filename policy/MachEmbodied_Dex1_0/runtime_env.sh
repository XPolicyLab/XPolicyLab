#!/bin/bash
set -euo pipefail

default_cache="${XDG_CACHE_HOME:-${HOME}/.cache}/mach_embodied_dex1_0"
runtime_path="${MACH_EMBODIED_DEX_RUNTIME_PATH:-${MACH_EMBODIED_DEX_SOURCE_DIR:-${MACH_EMBODIED_DEX_CACHE_DIR:-${default_cache}}/source}/runtime}"
if [[ ! -d "${runtime_path}" ]]; then
    echo "[ERROR] MachEmbodied-Dex runtime not found: ${runtime_path}. Run install.sh first." >&2
    return 2 2>/dev/null || exit 2
fi
export PYTHONPATH="${runtime_path}:${PYTHONPATH:-}"
unset default_cache runtime_path
export PYTHONNOUSERSITE=1
export WAN_DISABLE_FLASH_ATTN="${WAN_DISABLE_FLASH_ATTN:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
