#!/bin/bash
# Policy-side cuRobo for the vendored RoboDojo planner (auriga_xpl/planner.py).
#
#   bash install_planner.sh [python]        # default: python on PATH (the activated policy env)
#
# Installs, into that Python environment:
#   - PyTorch 2.7.0 + CUDA 12.8 wheels (the stack RoboDojo's scripts/install.sh installs), unless a
#     CUDA build of torch is already present;
#   - warp-lang 1.11.0 (RoboDojo's pin; cuRobo v2 JIT-compiles its kernels with warp);
#   - cuRobo v2 at RoboDojo's third_party/curobo pin d17b54ce, editable, with the [cu12] extra,
#     exactly as RoboDojo installs it (pip install -e ".[cu12]" --no-build-isolation). cuRobo v2 has
#     no mandatory compiled extension; its kernels are compiled by warp on first use and cached in
#     ~/.cache/warp (planner construction: 23 s with an empty cache, 9 s with a warm one, RTX 4090).
# Measured on an RTX 4090 host (Python 3.12.14, numpy 2.5.3, scipy 1.18.1): 28 min, almost all of it the 1.1 GB
# torch wheel download at 2.4 MB/s.
# Needs an NVIDIA driver for CUDA 12.8 (>= 570) and a GPU with compute capability >= 7.0.
# GPU memory: about 2.7 GB per process (one X5 planner shared by both arms; 1.9 GB torch allocated).
#
# Environment overrides: CUROBO_SRC (checkout dir, default third_party/curobo next to this script),
# TORCH_INDEX_URL (default https://download.pytorch.org/whl/cu128), PIP_EXTRA (extra pip options).
set -euo pipefail

PY=${1:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUROBO_REPO_URL="https://github.com/yuechen0614/curobo.git"   # RoboDojo .gitmodules
CUROBO_COMMIT="d17b54ce32cba095c0b000c4c58777075d11de0e"       # RoboDojo 726e9aa third_party/curobo
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
CUROBO_SRC=${CUROBO_SRC:-${SCRIPT_DIR}/third_party/curobo}
read -r -a PIP_EXTRA_ARGS <<< "${PIP_EXTRA:-}"

pip_install() { "${PY}" -m pip install "${PIP_EXTRA_ARGS[@]}" "$@"; }

echo "[install_planner] python: $("${PY}" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

if ! "${PY}" -c 'import torch, sys; sys.exit(0 if torch.version.cuda else 1)' >/dev/null 2>&1; then
    pip_install torch==2.7.0 --index-url "${TORCH_INDEX_URL}"
fi
pip_install warp-lang==1.11.0 setuptools_scm wheel

if [[ ! -d "${CUROBO_SRC}/.git" ]]; then
    mkdir -p "$(dirname "${CUROBO_SRC}")"
    git init -q "${CUROBO_SRC}"
    git -C "${CUROBO_SRC}" remote add origin "${CUROBO_REPO_URL}"
fi
if [[ "$(git -C "${CUROBO_SRC}" rev-parse HEAD 2>/dev/null || true)" != "${CUROBO_COMMIT}" ]]; then
    git -C "${CUROBO_SRC}" fetch --depth 1 origin "${CUROBO_COMMIT}"
    git -C "${CUROBO_SRC}" checkout -q --detach "${CUROBO_COMMIT}"
fi
# setuptools_scm needs a version when the checkout is shallow; RoboDojo's editable install reports 0.0.0.
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip_install -e "${CUROBO_SRC}[cu12]" --no-build-isolation
pip_install warp-lang==1.11.0   # keep RoboDojo's pin if the extra pulled a newer warp

"${PY}" - <<'PY'
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
import torch, warp
print("[install_planner] cuRobo importable; torch", torch.__version__, "cuda", torch.version.cuda,
      "warp", warp.__version__)
PY
