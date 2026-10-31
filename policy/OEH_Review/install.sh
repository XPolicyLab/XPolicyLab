#!/bin/bash
# OEH_Review installation: no weights of its own (OpenWAM's checkpoint: download_checkpoint.sh), no pip packages.
#   1. verify the vendored runtime against runtime/MANIFEST.json;
#   2. check the policy Python (the OpenWAM policy env: XPolicyLab, the vendored OpenWAM, torch, websockets, numpy,
#      Pillow, pyyaml) and, when given, the eval client's Python (numpy, Pillow, pyyaml);
#   3. build the OEH kernel from runtime/kernel/ with cargo (rustup.rs; a C compiler for the TLS crates).
# Both Python environments use the runtime from runtime/ through PYTHONPATH (the eval scripts set it); nothing is
# installed into them.
set -euo pipefail

usage() {
    echo "usage: bash install.sh --policy-python /path/to/openwam-env/bin/python [--eval-python /path/to/python]" >&2
    echo "                       [--cargo /path/to/cargo] [--skip-kernel]" >&2
}

policy_python=""
eval_python=""
cargo_bin="${CARGO:-cargo}"
skip_kernel=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-python) policy_python="${2:?--policy-python needs a path}"; shift 2 ;;
        --eval-python) eval_python="${2:?--eval-python needs a path}"; shift 2 ;;
        --cargo) cargo_bin="${2:?--cargo needs a path}"; shift 2 ;;
        --skip-kernel) skip_kernel=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
if [[ -z "${policy_python}" || ! -x "${policy_python}" ]]; then
    usage
    echo "[INSTALL][ERROR] --policy-python must be the OpenWAM policy environment's Python (policy/OpenWAM/README.md)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
OPENWAM_ROOT="${OPENWAM_ROOT:-${XPL_ROOT}/policy/OpenWAM/OpenWAM}"
KERNEL_SRC="${SCRIPT_DIR}/runtime/kernel"

echo "[INSTALL] 1/3 vendored runtime"
"${policy_python}" "${SCRIPT_DIR}/tools/verify_runtime.py"

echo "[INSTALL] 2/3 Python environments"
PYTHONPATH="${SCRIPT_DIR}/runtime:${BENCH_ROOT}:${OPENWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${policy_python}" "${SCRIPT_DIR}/tools/check_deps.py" policy
if [[ -n "${eval_python}" ]]; then
    PYTHONPATH="${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}" "${eval_python}" "${SCRIPT_DIR}/tools/check_deps.py" eval
fi

if [[ -n "${skip_kernel}" ]]; then
    echo "[INSTALL] 3/3 kernel build skipped (--skip-kernel); the frozen configuration needs ${KERNEL_SRC}/target/release/oeh"
else
    echo "[INSTALL] 3/3 OEH kernel (cargo build --release -p oeh-cli --locked --features oeh-brain/tls)"
    if ! command -v "${cargo_bin}" >/dev/null 2>&1; then
        echo "[INSTALL][ERROR] cargo not found (${cargo_bin}); install Rust from https://rustup.rs or pass --cargo" >&2
        exit 1
    fi
    (cd "${KERNEL_SRC}" && "${cargo_bin}" build --release -p oeh-cli --locked --features "${OEH_REVIEW_KERNEL_FEATURES:-oeh-brain/tls}")
    kernel="${KERNEL_SRC}/target/release/oeh"
    [[ -x "${kernel}" ]] || { echo "[INSTALL][ERROR] kernel build produced no ${kernel}" >&2; exit 1; }
    mkdir -p "${SCRIPT_DIR}/state"
    "${policy_python}" - "${kernel}" "${SCRIPT_DIR}/state/kernel.sha256" <<'PY'
import hashlib, sys
digest = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
open(sys.argv[2], "w").write(digest + "  " + sys.argv[1] + "\n")
print(f"[INSTALL] kernel sha256 {digest}")
PY
fi

echo "[INSTALL] OEH_Review is ready. Next: bash download_checkpoint.sh (OpenWAM-Alpha-Sim-RoboDojo), then eval.sh;"
echo "[INSTALL] the review configuration also needs the API key file (README, Configuration)."
