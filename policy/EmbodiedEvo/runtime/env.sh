#!/usr/bin/env bash
# Shared by entrypoints; accepts a venv directory, Python executable or conda env.
activate_policy_env() {
    local requested=${1:?environment required}
    if [[ -x "$requested/bin/python" ]]; then
        export PATH="$(cd "$requested/bin" && pwd):$PATH"
    elif [[ -x "$requested" && -f "$requested" ]]; then
        export PATH="$(cd "$(dirname "$requested")" && pwd):$PATH"
    elif [[ "$requested" != current ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$requested"
    fi
    export PYTHONPATH="${XPL_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONDONTWRITEBYTECODE=1
}
