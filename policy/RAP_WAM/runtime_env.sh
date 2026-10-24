#!/usr/bin/env bash
# Source from every entry point: writable runtime state belongs to this workspace.
# Existing model assets and interpreter binaries may be read from other locations.

: "${RAP_WAM_WORK_DIR:?Set RAP_WAM_WORK_DIR to an absolute workspace on shared storage}"
RAP_WAM_BOOTSTRAP_PYTHON="${RAP_WAM_BOOTSTRAP_PYTHON:-${PYTHON:-python3}}"
export RAP_WAM_BOOTSTRAP_PYTHON PYTHONDONTWRITEBYTECODE=1

rap_wam_require_work_path() {
    "$RAP_WAM_BOOTSTRAP_PYTHON" -B - "$RAP_WAM_WORK_DIR" "$1" "${2:-output}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = Path(sys.argv[2])
if not root.is_absolute() or not path.is_absolute():
    raise SystemExit(f"{sys.argv[3]} and RAP_WAM_WORK_DIR must be absolute paths")
root, path = root.resolve(), path.resolve()
if root == Path(root.anchor):
    raise SystemExit("RAP_WAM_WORK_DIR must be a dedicated workspace, not a filesystem root")
try:
    path.relative_to(root)
except ValueError:
    raise SystemExit(f"{sys.argv[3]} escapes RAP_WAM_WORK_DIR after resolving symlinks: {path}")
print(path)
PY
}

RAP_WAM_WORK_DIR="$(rap_wam_require_work_path "$RAP_WAM_WORK_DIR" workspace)" || return 2
export RAP_WAM_WORK_DIR
RAP_WAM_ENV="$(rap_wam_require_work_path "${RAP_WAM_ENV:-$RAP_WAM_WORK_DIR/env}" environment)" || return 2
export RAP_WAM_ENV

# Replace inherited cache paths rather than letting a caller's home/system cache leak in.
export TMPDIR="$RAP_WAM_WORK_DIR/tmp" TMP="$RAP_WAM_WORK_DIR/tmp" TEMP="$RAP_WAM_WORK_DIR/tmp"
export XDG_CACHE_HOME="$RAP_WAM_WORK_DIR/cache/xdg" XDG_CONFIG_HOME="$RAP_WAM_WORK_DIR/config/xdg"
export XDG_DATA_HOME="$RAP_WAM_WORK_DIR/data/xdg" XDG_STATE_HOME="$RAP_WAM_WORK_DIR/state/xdg"
export PIP_CACHE_DIR="$RAP_WAM_WORK_DIR/cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_CONFIG_FILE=/dev/null
export HF_HOME="$RAP_WAM_WORK_DIR/cache/hf" HF_HUB_CACHE="$RAP_WAM_WORK_DIR/cache/hf/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" HF_ASSETS_CACHE="$RAP_WAM_WORK_DIR/cache/hf/assets"
export HF_XET_CACHE="$RAP_WAM_WORK_DIR/cache/hf/xet" TRANSFORMERS_CACHE="$RAP_WAM_WORK_DIR/cache/hf/transformers"
export TORCH_HOME="$RAP_WAM_WORK_DIR/cache/torch" TORCH_EXTENSIONS_DIR="$RAP_WAM_WORK_DIR/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RAP_WAM_WORK_DIR/cache/torchinductor" TRITON_CACHE_DIR="$RAP_WAM_WORK_DIR/cache/triton"
export CUDA_CACHE_PATH="$RAP_WAM_WORK_DIR/cache/cuda" NUMBA_CACHE_DIR="$RAP_WAM_WORK_DIR/cache/numba"
export MPLCONFIGDIR="$RAP_WAM_WORK_DIR/config/matplotlib" PYTHONUSERBASE="$RAP_WAM_WORK_DIR/python_user"
export UV_CACHE_DIR="$RAP_WAM_WORK_DIR/cache/uv" UV_PYTHON_INSTALL_DIR="$RAP_WAM_WORK_DIR/python"
export UV_TOOL_DIR="$RAP_WAM_WORK_DIR/uv_tools" CONDA_PKGS_DIRS="$RAP_WAM_WORK_DIR/cache/conda"
export CONDA_ENVS_PATH="$RAP_WAM_WORK_DIR/conda_envs" WANDB_DIR="$RAP_WAM_WORK_DIR/logs/wandb"
export WANDB_CACHE_DIR="$RAP_WAM_WORK_DIR/cache/wandb" WANDB_CONFIG_DIR="$RAP_WAM_WORK_DIR/config/wandb"

# Validate all paths before creating any of them, including pre-existing nested symlinks.
"$RAP_WAM_BOOTSTRAP_PYTHON" -B - <<'PY' || return 2
import os
from pathlib import Path
import tempfile

root = Path(os.environ["RAP_WAM_WORK_DIR"])
names = (
    "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
    "PIP_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE",
    "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "PYTHONUSERBASE",
    "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_TOOL_DIR", "CONDA_PKGS_DIRS", "CONDA_ENVS_PATH",
    "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
)
paths = [Path(os.environ[name]) for name in names] + [root / "logs", root / "run"]
for path in paths:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        raise SystemExit(f"Runtime path escapes RAP_WAM_WORK_DIR: {path} -> {path.resolve()}")
for path in paths:
    path.mkdir(parents=True, exist_ok=True)
    path.resolve().relative_to(root)
with tempfile.NamedTemporaryFile(prefix=".rap-wam-storage-check-", dir=os.environ["TMPDIR"]) as handle:
    Path(handle.name).resolve().relative_to(root)
PY

rap_wam_python_for_env() {
    local requested="$1"
    if [[ -x "$requested/bin/python" ]]; then
        printf '%s\n' "$requested/bin/python"
    elif [[ -f "$requested" && -x "$requested" ]]; then
        printf '%s\n' "$requested"
    elif [[ "$requested" == "system" ]]; then
        command -v "${RAP_WAM_DEBUG_PYTHON:-python3}"
    else
        (
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "$requested"
            command -v python
        )
    fi
}
