#!/bin/bash
# Mibot-RSI holds no network weights. Prepare the Python of the RoboDojo evaluation client:
#   1. check the harness's Python dependencies (numpy, scipy, Pillow, opencv, pyyaml); --install-deps installs
#      missing ones within the supported ranges;
#   2. verify the harness runtime in runtime/ (an eval-only export of one harness commit) against its EXPORT.json;
#   3. with --policy-python, XPolicyLab itself (pip install -e <XPolicyLab root>) into the policy server's Python:
#      the server serves no model but needs XPolicyLab's dependencies (websockets>=14, which the RoboDojo
#      environment does not have).
# Knowledge snapshots are downloaded separately: download_checkpoint.sh.
set -euo pipefail

usage() {
    echo "usage: bash install.sh [--install-deps] [--policy-python /path/to/policy-env/python] [/path/to/eval-env/python]" >&2
}

install_deps=""
python_bin="python3"
policy_python=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-deps) install_deps="--install"; shift ;;
        --policy-python) policy_python="${2:?--policy-python needs the Python of the policy environment}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) python_bin="$1"; shift ;;
    esac
done
if [[ ! -x "${python_bin}" ]]; then
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        echo "[INSTALL][ERROR] Python not found: ${python_bin}" >&2
        exit 1
    fi
    python_bin="$(command -v "${python_bin}")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${python_bin}" "${SCRIPT_DIR}/artifacts.py" deps ${install_deps}
"${python_bin}" "${SCRIPT_DIR}/artifacts.py" check
if [[ -n "${policy_python}" ]]; then
    "${policy_python}" -m pip install -e "$(cd "${SCRIPT_DIR}/../.." && pwd)"
    echo "[INSTALL] policy server environment: ${policy_python}"
fi
echo "[INSTALL] Mibot-RSI is ready for ${python_bin}. Next: bash download_checkpoint.sh <ckpt_name> (README, Model Assets)."
