#!/usr/bin/env bash
# Sourced only by this adapter. Activate an explicit venv directory or conda name.
activate_python() {
    local requested=${1:?Python environment path or conda name is required}
    if [[ -x "${requested}/bin/python" ]]; then
        export PATH="$(cd "${requested}" && pwd)/bin:${PATH}"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${requested}"
    else
        echo "[ERROR] No Python environment at '${requested}' and conda is unavailable" >&2
        return 1
    fi
}
validate_mode() {
    case "${EVAL_ENV_TYPE:-sim}:${EMBODIEDEVO_INTERFACE_PROBE:-0}" in
        debug:1|sim:0|:0) ;;
        *) echo '[ERROR] Use sim without a probe, or debug with EMBODIEDEVO_INTERFACE_PROBE=1' >&2; return 1 ;;
    esac
}
