#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
if [[ ! -f "$XPL_ROOT/setup_policy_server.py" ]]; then
    echo 'Install this directory at XPolicyLab/policy/RSIWAM first.' >&2
    exit 1
fi
# The caller selects a dedicated policy environment and installs the matching
# CUDA torch/torchvision pair first; no simulator environment is changed here.
if [[ -n "${RSIWAM_CONSTRAINTS:-}" && ( ! -f "$RSIWAM_CONSTRAINTS" || ! -r "$RSIWAM_CONSTRAINTS" ) ]]; then
    echo 'RSIWAM_CONSTRAINTS must name a readable constraints file.' >&2
    exit 1
fi

torch_constraints="$(mktemp "${TMPDIR:-/tmp}/rsiwam-constraints.XXXXXX")"
trap 'rm -f -- "$torch_constraints"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python - "$torch_constraints" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

versions = {}
for package in ("torch", "torchvision"):
    try:
        versions[package] = version(package)
    except PackageNotFoundError:
        raise SystemExit(
            f"RSIWAM requires preinstalled {package}. Install a compatible CUDA "
            "torch/torchvision pair in the selected policy environment before install.sh."
        ) from None
Path(sys.argv[1]).write_text(
    "".join(f"{package}=={installed}\n" for package, installed in versions.items()),
    encoding="utf-8",
)
PY

# Keep the policy's dependency bounds while installing XPolicyLab too. Its
# transitive dependencies must not replace an already resolved policy package.
constraint_args=(-c "$torch_constraints" -c "$SCRIPT_DIR/bundle/requirements.txt")
if [[ -n "${RSIWAM_CONSTRAINTS:-}" ]]; then
    constraint_args+=(-c "$RSIWAM_CONSTRAINTS")
fi
python -m pip install -r "$SCRIPT_DIR/bundle/requirements.txt" "${constraint_args[@]}"
python -m pip install --no-deps -e "$SCRIPT_DIR" "${constraint_args[@]}"
python -m pip install -e "$XPL_ROOT" "${constraint_args[@]}"
printf '%s\n' 'RSIWAM installed in the active policy environment; the preinstalled torch/torchvision versions were constrained throughout installation.'
