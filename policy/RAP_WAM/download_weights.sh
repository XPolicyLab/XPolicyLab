#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$DIR/../.." && pwd)"
source "$DIR/runtime_env.sh"
# Validate an explicit destination; Python also checks its workspace-local default.
output=""
args=("$@")
for ((index=0; index<${#args[@]}; index++)); do
    case "${args[index]}" in
        --output) output="${args[index+1]:-}" ;;
        --output=*) output="${args[index]#--output=}" ;;
    esac
done
if [[ -n "$output" ]]; then
    rap_wam_require_work_path "$output" 'inference bundle output' >/dev/null
fi
export PYTHONPATH="$(dirname "$XPL_ROOT"):$XPL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${RAP_WAM_PYTHON:-$RAP_WAM_ENV/bin/python}" -m XPolicyLab.policy.RAP_WAM.download_weights "$@"
