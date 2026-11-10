#!/usr/bin/env bash
# Create the HumanWAM-Goal policy environment: the released Open-HumanWAM code at the
# pinned revision, its humanwam_goal environment (Python 3.13, PyTorch 2.10, CUDA 12.8,
# the pinned cuBLASLt) and XPolicyLab.
#   bash install.sh               # install
#   bash install.sh --check-only  # print the plan; change nothing
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
echo "[INSTALL] HWGOAL_REPO=${HWGOAL_REPO} (release ${repository}@${revision}), environment ${environment}"
if [[ "${1:-}" == --check-only ]]; then
    test -f "${HWGOAL_XPL_ROOT}/pyproject.toml"
    [[ -f "${package}/pyproject.toml" ]] || echo "[INSTALL] code is not checked out yet"
    [[ -x "${environment}/bin/python" ]] || echo "[INSTALL] the policy environment does not exist yet"
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
    # Installs the package with exact pins and the cuBLASLt 12.8.3.14 library that
    # the published results were served with (environment/runtime/cublaslt-12.8.3.14).
    bash "${package}/scripts/setup_environment.sh" "${environment}"
fi
uv pip install --python "${environment}/bin/python" -e "${HWGOAL_XPL_ROOT}"
echo "[INSTALL] policy environment: ${environment} (pass it as policy_conda_env)"
