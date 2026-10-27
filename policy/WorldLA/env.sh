#!/usr/bin/env bash
# Source only: use an existing environment, never install into it implicitly.
worldla_activate_env() {
    local selector=${1:?environment name, directory or Python executable required}
    if [[ -f "$selector" && -x "$selector" ]]; then
        export PATH="$(dirname "$selector"):$PATH"
        WORLDLA_PYTHON="$selector"
    elif [[ -x "$selector/bin/python" ]]; then
        export PATH="$selector/bin:$PATH"
        WORLDLA_PYTHON="$selector/bin/python"
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "Conda unavailable; pass an absolute environment/Python path." >&2
            return 1
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$selector"
        WORLDLA_PYTHON="$(command -v python)"
    fi
    export WORLDLA_PYTHON
}
