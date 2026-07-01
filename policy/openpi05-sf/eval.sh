#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_REAL="$(readlink -f "${POLICY_DIR}")"
CONFIG_PATH="${POLICY_DIR}/deploy.yml"
DEFAULT_XPL_ROOT=""

if [[ -z "${XPL_ROOT:-}" && -d "${DEFAULT_XPL_ROOT}" ]]; then
    XPL_ROOT="${DEFAULT_XPL_ROOT}"
fi

if [[ -z "${XPL_ROOT:-}" ]]; then
    echo "[ERROR] XPL_ROOT must point to an XPolicyLab checkout." >&2
    echo "[ERROR] Example: XPL_ROOT=/path/to/XPolicyLab bash ${BASH_SOURCE[0]}" >&2
    exit 2
fi

XPL_ROOT="$(cd "${XPL_ROOT}" && pwd)"
XPL_PARENT="$(dirname "${XPL_ROOT}")"
SETUP_POLICY_SERVER="${XPL_ROOT}/setup_policy_server.py"
POLICY_LINK="${XPL_ROOT}/policy/Pi_05_SF"

if [[ ! -f "${SETUP_POLICY_SERVER}" ]]; then
    echo "[ERROR] setup_policy_server.py not found under XPL_ROOT=${XPL_ROOT}" >&2
    exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] deploy.yml not found at ${CONFIG_PATH}" >&2
    exit 2
fi

if [[ -L "${POLICY_LINK}" ]]; then
    LINK_TARGET="$(readlink -f "${POLICY_LINK}")"
    if [[ "${LINK_TARGET}" != "${POLICY_REAL}" ]]; then
        echo "[ERROR] ${POLICY_LINK} points to ${LINK_TARGET}, not ${POLICY_REAL}" >&2
        exit 2
    fi
elif [[ -e "${POLICY_LINK}" ]]; then
    LINK_TARGET="$(readlink -f "${POLICY_LINK}")"
    if [[ "${LINK_TARGET}" != "${POLICY_REAL}" ]]; then
        echo "[ERROR] ${POLICY_LINK} exists at ${LINK_TARGET}, not ${POLICY_REAL}; refusing to overwrite." >&2
        exit 2
    fi
else
    mkdir -p "$(dirname "${POLICY_LINK}")"
    ln -s "${POLICY_REAL}" "${POLICY_LINK}"
fi

OPENPI_SRC="${POLICY_REAL}/openpi/src"
OPENPI_CLIENT_SRC="${POLICY_REAL}/openpi/packages/openpi-client/src"
VGGT_SRC="${POLICY_REAL}/openpi/src/vggt"
export PYTHONPATH="${XPL_PARENT}:${OPENPI_SRC}:${OPENPI_CLIENT_SRC}:${VGGT_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec python "${SETUP_POLICY_SERVER}" \
    --config_path "${CONFIG_PATH}" \
    "$@"
