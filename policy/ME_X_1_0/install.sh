#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEX_SOURCE_REPO="${MEX_SOURCE_REPO:-https://github.com/Liuxuetao1219/ME_X_1_0.git}"
MEX_SOURCE_REVISION="${MEX_SOURCE_REVISION:-main}"
MEX_CACHE_DIR="${MEX_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/me_x_1_0}"
MEX_SOURCE_DIR="${MEX_SOURCE_DIR:-${MEX_CACHE_DIR}/source}"

mkdir -p "${MEX_CACHE_DIR}"
if [[ ! -d "${MEX_SOURCE_DIR}/.git" ]]; then
    git clone "${MEX_SOURCE_REPO}" "${MEX_SOURCE_DIR}"
fi
git -C "${MEX_SOURCE_DIR}" fetch --depth 1 origin "${MEX_SOURCE_REVISION}"
git -C "${MEX_SOURCE_DIR}" checkout --detach FETCH_HEAD

python -m pip install -e "${XPL_ROOT}"
python -m pip install -r "${MEX_SOURCE_DIR}/runtime/requirements.txt"
