#!/usr/bin/env bash

# Source this file, then call starvla_activate_policy_env ENV_REF.  ENV_REF may
# be a virtualenv directory, a uv project containing .venv, a Python
# executable, or a Conda environment name.  The function exports the exact
# interpreter as STARVLA_POLICY_PYTHON.
starvla_activate_policy_env() {
    local env_ref=${1:?StarVLA policy environment is required}
    local env_dir=""

    if [[ -x "${env_ref}" && ! -d "${env_ref}" ]]; then
        STARVLA_POLICY_PYTHON="${env_ref}"
        export STARVLA_POLICY_PYTHON
        echo "[starVLA env] Python executable: ${STARVLA_POLICY_PYTHON}" >&2
        return 0
    fi

    if [[ -x "${env_ref}/bin/python" ]]; then
        env_dir="$(cd "${env_ref}" && pwd -L)"
        if [[ -f "${env_dir}/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source "${env_dir}/bin/activate"
        fi
        STARVLA_POLICY_PYTHON="${env_dir}/bin/python"
        export STARVLA_POLICY_PYTHON
        echo "[starVLA env] Virtual environment: ${env_dir}" >&2
        return 0
    fi

    if [[ -x "${env_ref}/.venv/bin/python" ]]; then
        env_dir="$(cd "${env_ref}/.venv" && pwd -L)"
        # shellcheck disable=SC1091
        source "${env_dir}/bin/activate"
        STARVLA_POLICY_PYTHON="${env_dir}/bin/python"
        export STARVLA_POLICY_PYTHON
        echo "[starVLA env] uv virtual environment: ${env_dir}" >&2
        return 0
    fi

    if ! command -v conda >/dev/null 2>&1; then
        echo "[starVLA env][ERROR] '${env_ref}' is not a Python/venv path and Conda is unavailable." >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env_ref}"
    STARVLA_POLICY_PYTHON="$(command -v python)"
    export STARVLA_POLICY_PYTHON
    echo "[starVLA env] Conda environment: ${env_ref} (${STARVLA_POLICY_PYTHON})" >&2
}
