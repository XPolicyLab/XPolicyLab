#!/usr/bin/env bash
# Shared interpreter selection; sourced by adapter entry points.
set -euo pipefail
SLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SLA_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

sla_python() {
    local env_path="${1:-uv}"
    if [[ "${env_path}" == uv ]]; then
        env_path="${SLA_DIR}/openpi"
    elif [[ "${env_path}" != /* ]]; then
        env_path="${SLA_DIR}/${env_path}"
    fi
    PYTHON_BIN="${SLA_PYTHON:-${env_path}/.venv/bin/python}"
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "[SLA] Python not found: ${PYTHON_BIN}. Run bash install.sh or select a uv project." >&2
        return 1
    fi
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
    # Parent imports XPolicyLab; checkout also supports the shared transport's client_server imports.
    export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${SLA_DIR}/openpi/src:${SLA_DIR}/openpi/packages/openpi-client/src:${SLA_DIR}/openpi${PYTHONPATH:+:${PYTHONPATH}}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export PYTHONUNBUFFERED=1
}
