#!/usr/bin/env bash
# Set up the HumanWAM-Goal policy environment from the pinned Open-HumanWAM revision.
#   bash install.sh               # install code + env/
#   bash install.sh --check-only  # verify layout without modifying disk
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

read -r repository revision subdirectory < <(python3 - "${HWGOAL_ADAPTER_DIR}/release.json" <<'PY'
import json, sys
code = json.load(open(sys.argv[1]))["code"]
print(code.get("repository") or "-", code.get("revision") or "-", code["subdirectory"])
PY
)
environment=${HWGOAL_ENV:-${HWGOAL_ADAPTER_DIR}/env}
package="${HWGOAL_REPO}/${subdirectory}"
echo "[INSTALL] HWGOAL_REPO=${HWGOAL_REPO} (${repository}@${revision}), environment=${environment}"
if [[ "${1:-}" == --check-only ]]; then
    test -f "${HWGOAL_XPL_ROOT}/pyproject.toml"
    [[ -f "${package}/pyproject.toml" ]] || echo "[INSTALL] code is not checked out yet"
    [[ -x "${environment}/bin/python" ]] || echo "[INSTALL] policy environment does not exist yet"
    exit 0
fi

if [[ ! -f "${package}/pyproject.toml" ]]; then
    if [[ "${repository}" == - || "${revision}" == - ]]; then
        echo "[INSTALL][ERROR] release.json pins no code revision; set HWGOAL_REPO to an Open-HumanWAM checkout" >&2
        exit 2
    fi
    git clone "${repository}" "${HWGOAL_REPO}"
    git -C "${HWGOAL_REPO}" checkout --detach "${revision}"
fi
command -v uv >/dev/null || { echo "[INSTALL][ERROR] uv is required: https://docs.astral.sh/uv/" >&2; exit 2; }
if [[ ! -x "${environment}/bin/python" ]]; then
    bash "${package}/scripts/setup_environment.sh" "${environment}"
fi
if [[ -f "${package}/posttrain/pyproject.toml" ]]; then
    uv pip install --python "${environment}/bin/python" -e "${package}/posttrain[convert]"
fi
uv pip install --python "${environment}/bin/python" -e "${HWGOAL_XPL_ROOT}"
echo "[INSTALL] policy environment ready: ${environment}"
