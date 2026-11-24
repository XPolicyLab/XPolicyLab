#!/bin/bash
# The policy environment of PyRUA_Lean: XPolicyLab's policy server, PyRUA-Lean, and SAM3, the
# one model the policy side loads.  Run it inside the environment the policy server will use
# (<policy_env> of eval.sh).  That environment needs a CUDA build of torch and the torchvision
# that goes with it first: this script installs neither, and keeps both as they are.
#
#   bash install.sh [<PyRUA-Lean checkout or pip requirement>]
#
# Without an argument PyRUA-Lean is installed from the checkout this directory is linked from
# (integrations/xpolicylab/PyRUA_Lean of a PyRUA-Lean checkout, linked into XPolicyLab/policy),
# and from the public repository, at the revision below, when the directory was copied instead.
# The simulator's environment needs nothing beyond RoboDojo's own installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKOUT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)"
if [[ -f "${CHECKOUT}/src/pyrualean/__init__.py" ]]; then
    DEFAULT_SOURCE="${CHECKOUT}"
else
    # The revision of PyRUA-Lean the adapter was evaluated with.
    DEFAULT_SOURCE="git+https://github.com/DAGroup-PKU/PyRUA-Lean.git@c6861200bfb190169953d1827260ccf71320eea1"
fi
PYRUALEAN_SOURCE="${1:-${PYRUALEAN_SOURCE:-${DEFAULT_SOURCE}}}"
# The revision of the sam3 package the masks were measured with.
SAM3_SOURCE="${SAM3_SOURCE:-git+https://github.com/RLinf/sam3.git@c1f6b1e83967c912f675d8975d437ccdc41e17d8}"

if ! python -c "import torch, torchvision, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[install] this environment needs a CUDA build of torch, and the torchvision built for it," >&2
    echo "[install] before SAM3 can load (see https://pytorch.org/get-started/locally/)" >&2
    exit 1
fi

# pip must leave that torch alone: SAM3's requirements would otherwise pull in the newest torch
# and torchvision, built for whatever CUDA is current.  SAM3 also needs numpy 1.x, and with it
# the last OpenCV line that still supports numpy 1.x.
constraints="$(mktemp)"
trap 'rm -f "${constraints}"' EXIT
python - > "${constraints}" <<'PY'
import importlib.metadata as md

for name in ("torch", "torchvision"):
    print(f"{name}=={md.version(name)}")
print("numpy>=1.26,<2")
print("opencv-python-headless<4.12")
PY
pip_install() { python -m pip install -c "${constraints}" "$@"; }

# XPolicyLab itself: the websocket policy server and the image decoding it does.
pip_install -e "${XPL_ROOT}"
if [[ -d "${PYRUALEAN_SOURCE}" ]]; then
    pip_install -e "${PYRUALEAN_SOURCE}"
else
    pip_install "pyrualean @ ${PYRUALEAN_SOURCE}"
fi
# What PyRUA-Lean's episode driver imports (it declares no dependencies of its own).
pip_install "numpy" "pillow" "pyyaml>=6" "httpx>=0.27" "mcp>=1.8,<2" "uvicorn"
# SAM3, and what its image model imports without declaring it.
pip_install "sam3 @ ${SAM3_SOURCE}" "einops" "pycocotools" "psutil"

python - <<'PY'
import torch
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
from sam3.model_builder import build_sam3_image_model  # noqa: F401

import pyrualean.robots

pyrualean.robots.get_adapter("robodojo_xpl_rgb").host_module()
print(f"[install] torch {torch.__version__}, SAM3 and PyRUA-Lean import")
PY

echo "[install] done.  SAM3's checkpoint: bash ${SCRIPT_DIR}/download_assets.sh"
