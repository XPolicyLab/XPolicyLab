#!/bin/bash
# The OpenWAM-Alpha RoboDojo checkpoint at its pinned Hugging Face revision (configs/openwam-robodojo-pins.json:
# repo, revision, every file's size and sha256), into the official OpenWAM adapter's checkpoint root, where
# policy/OpenWAM resolves it as well. OEH_Review freezes the directory's identity into its configuration; a changed
# file is refused at server start.
#   usage: bash download_checkpoint.sh [/path/to/python]   (needs huggingface_hub; OPENWAM_CKPT_ROOT overrides the root)
set -euo pipefail

python_bin="${1:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CKPT_ROOT="${OPENWAM_CKPT_ROOT:-${XPL_ROOT}/policy/OpenWAM/checkpoints}"

"${python_bin}" -c "import huggingface_hub" 2>/dev/null || {
    echo "[DOWNLOAD][ERROR] ${python_bin} has no huggingface_hub (pip install huggingface_hub)" >&2
    exit 1
}
"${python_bin}" "${SCRIPT_DIR}/tools/download_openwam_pins.py" "${SCRIPT_DIR}/configs/openwam-robodojo-pins.json" "${CKPT_ROOT}"
echo "[DOWNLOAD] checkpoint at ${CKPT_ROOT}/OpenWAM-Alpha-Sim-RoboDojo (manifest: OpenWAM-Alpha-Sim-RoboDojo.observed.json)"
