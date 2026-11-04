#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIN=470278c478bdb5e4b6f17dde1ea0393761f230f0
if [[ ${1:-} == --interface-only ]]; then
    [[ $# -eq 2 ]] || { echo 'Usage: install.sh --interface-only /absolute/venv' >&2; exit 2; }
    TARGET_ENV=$2
else
    [[ $# -ge 2 && $# -le 3 ]] || { echo 'Usage: install.sh /absolute/venv /absolute/upstream_checkout [upstream_sha]' >&2; exit 2; }
    TARGET_ENV=$1
    UPSTREAM_DIR=$2
    [[ ${3:-$PIN} == "$PIN" ]] || { echo '[ERROR] This adapter is pinned to its declared revision' >&2; exit 2; }
    [[ ! -e "$UPSTREAM_DIR" ]] || { echo '[ERROR] Upstream checkout target must be new; existing source is preserved' >&2; exit 2; }
fi
[[ ! -e "$TARGET_ENV" ]] || { echo '[ERROR] Python environment target must be new' >&2; exit 2; }
python3 -m venv "$TARGET_ENV"
"$TARGET_ENV/bin/python" -m pip install -e "$XPL_ROOT"
if [[ ${1:-} != --interface-only ]]; then
    git clone --no-checkout https://github.com/Einsia/EmbodiedEvo.git "$UPSTREAM_DIR"
    git -C "$UPSTREAM_DIR" checkout --detach "$PIN"
    "$TARGET_ENV/bin/python" -m pip install -e "$UPSTREAM_DIR[policy]"
    echo 'Upstream installed. Provision runtime/assets using its docs; no model, Docker build or evaluation was started.'
else
    echo 'Interface dependencies installed; real evaluation assets/runtime were not provisioned.'
fi
