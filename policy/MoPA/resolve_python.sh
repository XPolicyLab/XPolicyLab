#!/bin/bash
# Resolve a named conda environment, an environment path, or an interpreter.
resolve_python() {
    local env_name="$1"
    if [[ -z "${env_name}" || "${env_name}" == "none" ]]; then
        command -v python
        return
    fi
    if [[ "${env_name}" == */* && -f "${env_name}" && -x "${env_name}" ]]; then
        echo "${env_name}"
        return
    fi
    if [[ -x "${env_name}/bin/python" ]]; then
        echo "${env_name}/bin/python"
        return
    fi
    local roots=() extra=()
    IFS=':' read -r -a extra <<<"${CONDA_ENVS_PATH:-}"
    roots+=("${extra[@]}" "${HOME}/.conda/envs")
    if command -v conda >/dev/null 2>&1; then
        roots+=("$(conda info --base 2>/dev/null)/envs")
    fi
    local root
    for root in "${roots[@]}"; do
        if [[ -n "${root}" && -x "${root}/${env_name}/bin/python" ]]; then
            echo "${root}/${env_name}/bin/python"
            return
        fi
    done
    echo "Cannot locate Python environment '${env_name}'; provide its full path or CONDA_ENVS_PATH." >&2
    return 1
}
