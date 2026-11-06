#!/usr/bin/env bash
# Shared by the HumanWAM-Goal entry scripts; sourcing it starts nothing.
set -euo pipefail

HWGOAL_ADAPTER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HWGOAL_XPL_ROOT=$(cd -- "${HWGOAL_ADAPTER_DIR}/../.." && pwd)
HWGOAL_BENCH_ROOT=$(cd -- "${HWGOAL_XPL_ROOT}/.." && pwd)
# install.sh checks out the released Open-HumanWAM code here unless HWGOAL_REPO is set;
# the policy environment installs its humanwam_goal package (import name goalwam).
HWGOAL_REPO=${HWGOAL_REPO:-${HWGOAL_ADAPTER_DIR}/OpenHumanWAM}
export HWGOAL_REPO
export PYTHONPATH="${HWGOAL_XPL_ROOT}:${HWGOAL_BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Accept a venv directory, a Python executable or a conda environment name.
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
