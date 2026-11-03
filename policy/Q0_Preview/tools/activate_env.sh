# Source with a Conda environment name or an absolute Python environment prefix.
if [[ -x "${1}/bin/python" ]]; then
    export PATH="${1}/bin:${PATH}"
    export Q0_PREVIEW_ACTIVE_PREFIX="$(cd "${1}" && pwd)"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${1}"
    export Q0_PREVIEW_ACTIVE_PREFIX="${CONDA_PREFIX}"
fi
