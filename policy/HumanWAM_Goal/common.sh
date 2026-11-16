#!/usr/bin/env bash
# Shared environment helpers for HumanWAM-Goal entry scripts.
set -euo pipefail

HWGOAL_ADAPTER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HWGOAL_XPL_ROOT=$(cd -- "${HWGOAL_ADAPTER_DIR}/../.." && pwd)
HWGOAL_BENCH_ROOT=$(cd -- "${HWGOAL_XPL_ROOT}/.." && pwd)
HWGOAL_REPO=${HWGOAL_REPO:-${HWGOAL_ADAPTER_DIR}/OpenHumanWAM}
export HWGOAL_REPO
export PYTHONPATH="${HWGOAL_XPL_ROOT}:${HWGOAL_BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure websockets >= 15 connects directly to local policy servers when HTTP(S)_PROXY is set.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1"

# Pick up --task_config from RoboTwin's multitask scheduler when HWGOAL_TASK_CONFIG is unset.
if [[ -z "${HWGOAL_TASK_CONFIG:-}" && -n "${ROBOTWIN_EVAL_ARGS_FILE:-}" && -f "${ROBOTWIN_EVAL_ARGS_FILE}" ]]; then
    _hwgoal_tc=$(awk '$0 == "--task_config" { getline; print; exit }' "${ROBOTWIN_EVAL_ARGS_FILE}")
    if [[ -n "${_hwgoal_tc}" ]]; then
        export HWGOAL_TASK_CONFIG="${_hwgoal_tc}"
    fi
    unset _hwgoal_tc
fi

# Accept a venv directory, a Python executable, or a conda environment name.
goalwam_select_python() {
    local specification=$1
    if [[ -x "${specification}/bin/python" ]]; then
        HWGOAL_EXEC_PYTHON="${specification}/bin/python"
    elif [[ -x "${specification}" && ! -d "${specification}" ]]; then
        HWGOAL_EXEC_PYTHON=${specification}
    else
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${specification}"
        HWGOAL_EXEC_PYTHON=$(command -v python)
    fi
}
