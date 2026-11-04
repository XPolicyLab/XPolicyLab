#!/bin/bash
set -euo pipefail

# Install XPolicyLab plus the vendored FlexWAM package into the active
# environment. Install PyTorch first (see the policy README) — this script
# does not pin a CUDA wheel.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FLEXWAM_ROOT="${FLEXWAM_ROOT:-${SCRIPT_DIR}/flexwam}"

compiler_ready() {
    command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1
}

# deepspeed compiles during the flexwam install. A compiler that is installed
# but not on PATH (common when /usr/bin was dropped) is picked up here. If
# none is installed, put gcc and g++ into the active conda env so they land
# on PATH without a system package install.
ensure_compiler() {
    if compiler_ready; then
        return 0
    fi
    local dir
    for dir in /usr/bin /usr/local/bin /bin; do
        if [[ -x "${dir}/gcc" && -x "${dir}/g++" ]]; then
            export PATH="${dir}:${PATH}"
            hash -r || true
            if compiler_ready; then
                echo "[install] using gcc/g++ from ${dir}"
                return 0
            fi
        fi
    done

    local conda_bin="${CONDA_EXE:-}"
    if [[ -z "${conda_bin}" ]]; then
        conda_bin="$(command -v conda 2>/dev/null || true)"
    fi
    if [[ -z "${conda_bin}" || -z "${CONDA_PREFIX:-}" ]]; then
        echo "[install] gcc and g++ are not available, and there is no active conda environment to install them into." >&2
        echo "[install] Install a C++ compiler and rerun. On Debian/Ubuntu: sudo apt install build-essential" >&2
        exit 1
    fi
    echo "[install] gcc/g++ not on PATH; installing them into ${CONDA_PREFIX}"
    "${conda_bin}" install -y -c conda-forge --freeze-installed gcc gxx
    hash -r || true
    if ! compiler_ready; then
        echo "[install] gcc/g++ were installed but are still not on PATH." >&2
        exit 1
    fi
}

ensure_compiler

python -m pip install -e "${XPL_ROOT}"
python -m pip install -e "${FLEXWAM_ROOT}"
