#!/usr/bin/env bash
# Shared local launch helpers; source from a policy entry script.

wsp_select_bench_root() {
    # Standard layout: <benchmark>/XPolicyLab. BENCH_ROOT is an explicit override.
    BENCH_ROOT="$(cd "${BENCH_ROOT:-${XPL_ROOT}/..}" && pwd)" || return
    export BENCH_ROOT
}

wsp_select_deploy() {
    local bench_name=$1
    if [[ "${bench_name,,}" != robodojo ]]; then
        echo "[CONFIG][ERROR] WorldScape_Policy_2.5-Preview supports RoboDojo only." >&2
        return 2
    fi
    local default_config=deploy.yml
    yaml_file="${WSP_DEPLOY_CONFIG:-${SCRIPT_DIR}/${default_config}}"
    [[ "${yaml_file}" = /* ]] || yaml_file="${SCRIPT_DIR}/${yaml_file}"
    if [[ ! -f "${yaml_file}" ]]; then
        echo "[CONFIG][ERROR] Missing deployment file: ${yaml_file}" >&2
        return 2
    fi
    export WSP_DEPLOY_CONFIG="${yaml_file}"
}

wsp_select_python() {
    local environment=$1
    local explicit_python=${2:-}
    if [[ -n "${explicit_python}" ]]; then
        WSP_SELECTED_PYTHON="${explicit_python}"
    elif [[ -x "${environment}/bin/python" ]]; then
        WSP_SELECTED_PYTHON="${environment}/bin/python"
    elif [[ -f "${environment}" && -x "${environment}" ]]; then
        WSP_SELECTED_PYTHON="${environment}"
    elif [[ "${environment}" == current ]]; then
        WSP_SELECTED_PYTHON="$(command -v python)"
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "[ENV][ERROR] Pass an environment path or set the Python override; conda is unavailable." >&2
            return 2
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${environment}"
        WSP_SELECTED_PYTHON="$(command -v python)"
    fi
    if [[ ! -x "${WSP_SELECTED_PYTHON}" ]]; then
        echo "[ENV][ERROR] Python is not executable: ${WSP_SELECTED_PYTHON}" >&2
        return 2
    fi
    export PATH="$(dirname "${WSP_SELECTED_PYTHON}"):${PATH}"
}

wsp_launch_or_print() {
    if [[ "${WSP_LAUNCH_DRY_RUN:-false}" == true ]]; then
        printf '[DRY-RUN] '
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    exec "$@"
}
