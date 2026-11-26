#!/bin/bash
# Sourced by the Auriga scripts. load_config_env exports the KEY=VALUE lines of config.env without
# overriding a variable the caller already set to a non-empty value, so an exported
# AURIGA_API_KEY wins over the empty line of a fresh config.env. Same rules as
# auriga_xpl/perception.py (_config_env): no shell expansion, surrounding quotes stripped.
load_config_env() {
    local file=$1 line key value
    [[ -f "${file}" ]] || return 0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"
        [[ "${line}" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]] || continue
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        if [[ "${value}" =~ ^\"(.*)\"$ || "${value}" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi
        [[ -n "${!key:-}" ]] && continue
        export "${key}=${value}"
    done < "${file}"
}
