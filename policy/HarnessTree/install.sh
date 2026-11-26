#!/usr/bin/env bash
# Adapter-side deps for the policy server process. Simulator / Isaac / cuRobo
# come from the RoboDojo eval environment (document in README). SAM3 weights:
# bash download_checkpoint.sh
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "[INSTALL][ERROR] usage: bash install.sh /path/to/python" >&2
  exit 1
fi
python_bin=$1
if [[ ! -x "${python_bin}" ]]; then
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[INSTALL][ERROR] Python not found: ${python_bin}" >&2
    exit 1
  fi
  python_bin="$(command -v "${python_bin}")"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Editable XPolicyLab so XPolicyLab.policy.HarnessTree imports resolve.
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${python_bin}" -e "${XPL_ROOT}"
  uv pip install --python "${python_bin}" "pillow>=10" "numpy>=1.24" "opencv-python-headless>=4.8"
else
  "${python_bin}" -m pip install -e "${XPL_ROOT}"
  "${python_bin}" -m pip install "pillow>=10" "numpy>=1.24" "opencv-python-headless>=4.8"
fi

echo "[INSTALL] HarnessTree client deps installed into ${python_bin}."
echo "[INSTALL] For sim eval: use the RoboDojo/Isaac eval env as arg 10 of eval.sh."
echo "[INSTALL] For SAM3: bash download_checkpoint.sh ${python_bin}"
echo "[INSTALL]   then run scripts/serve_sam3.py in an env with torch + the sam3 package"
echo "[INSTALL]   (RLinf/sam3 @ c1f6b1e83967c912f675d8975d437ccdc41e17d8 recommended)."
