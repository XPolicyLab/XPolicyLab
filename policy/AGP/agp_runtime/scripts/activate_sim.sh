#!/usr/bin/env bash
# The simulator runtime (sourced). MIGRATION_ROOT is the tree holding this script: the development tree W, or the
# packaged runtime XPolicyLab/policy/AGP/agp_runtime (scripts/package_policy.py). AGP_HOME (agp_core/paths.py) holds what
# the runtime uses and writes: .venv-sim, the Kit/CUDA/warp/torch caches, TMPDIR; unset it is MIGRATION_ROOT in the
# development tree and ~/.agp-robodojo when packaged (never inside the checkout).
MIGRATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MIGRATION_ROOT
if [[ -z "${AGP_HOME:-}" ]]; then
  if [[ "$(basename "$MIGRATION_ROOT")" == "agp_runtime" && -f "$MIGRATION_ROOT/../deploy.yml" ]]; then
    AGP_HOME="$HOME/.agp-robodojo"
  else
    AGP_HOME="$MIGRATION_ROOT"
  fi
fi
export AGP_HOME
export PATH="$AGP_HOME/.venv-sim/bin:$PATH"
export PYTHONPATH="$MIGRATION_ROOT:$MIGRATION_ROOT/source/RoboDojo:$MIGRATION_ROOT/source/RoboDojo/XPolicyLab"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export XDG_CACHE_HOME="$AGP_HOME/cache" XDG_DATA_HOME="$AGP_HOME/data" XDG_CONFIG_HOME="$AGP_HOME/config"
export TORCH_EXTENSIONS_DIR="$AGP_HOME/cache/torch_extensions" CUDA_CACHE_PATH="$AGP_HOME/cache/cuda"
# System libraries unpacked without root: AGP_RUNTIME_LIBS, else robodojo/runtime-libs
# next to AGP_HOME; prepended only when the folder exists (with the libraries installed system-wide nothing is needed)
_runtime_libs="${AGP_RUNTIME_LIBS:-$(cd "$AGP_HOME/.." 2>/dev/null && pwd -P || dirname "$AGP_HOME")/robodojo/runtime-libs}"
if [ -d "$_runtime_libs/usr/lib/x86_64-linux-gnu" ]; then
  export LD_LIBRARY_PATH="$_runtime_libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _runtime_libs
_vk_icd=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ ! -f "$_vk_icd" ] && [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  _vk_icd=/etc/vulkan/icd.d/nvidia_icd.json
fi
export VK_ICD_FILENAMES="$_vk_icd" VK_DRIVER_FILES="$_vk_icd" TERM=xterm-256color
unset _vk_icd

export OPTIX_CACHE_PATH="$AGP_HOME/cache/optix"
export TMPDIR="$AGP_HOME/tmp"
mkdir -p "$TMPDIR" 2>/dev/null || true
