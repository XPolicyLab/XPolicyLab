#!/usr/bin/env bash
set -euo pipefail
resume=0
if [[ $# -eq 1 && "$1" == "--resume" ]]; then
    resume=1
elif [[ $# -ne 0 ]]; then
    echo 'Usage: install.sh [--resume]' >&2
    exit 2
fi
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$DIR/../.." && pwd)"
source "$DIR/runtime_env.sh"
# Editable installation writes package metadata beside the source checkout.
rap_wam_require_work_path "$XPL_ROOT" 'XPolicyLab checkout for editable installation' >/dev/null
# pip's environment options override the selected venv even with config files
# disabled. Keep installation and incidental writes in this workspace; network
# indexes, proxies and certificate settings remain available to the caller.
unset PIP_TARGET PIP_PREFIX PIP_ROOT PIP_USER PIP_PYTHON
unset PIP_LOG PIP_SRC PIP_BUILD_TRACKER
if [[ -n "${PIP_REPORT:-}" && "$PIP_REPORT" != "-" ]]; then
    PIP_REPORT="$(rap_wam_require_work_path "$PIP_REPORT" 'pip report')"
    export PIP_REPORT
fi
py="${PYTHON:-python3}"
venv="$RAP_WAM_ENV"
if (( resume )); then
    "$py" -B - "$venv" <<'PY'
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
cfg = root / "pyvenv.cfg"
if not cfg.is_file():
    raise SystemExit("--resume requires an existing Python venv with pyvenv.cfg")
values = dict(line.split("=", 1) for line in cfg.read_text().splitlines() if "=" in line)
values = {key.strip().lower(): value.strip() for key, value in values.items()}
if values.get("include-system-site-packages", "").lower() != "false":
    raise SystemExit("--resume requires include-system-site-packages=false")
if not (root / "bin" / "python").is_file():
    raise SystemExit("--resume requires an existing venv interpreter")
if not (3, 10) <= sys.version_info[:2] < (3, 12):
    raise SystemExit("Use Python 3.10 or 3.11")
if values.get("version", "").split(".")[:2] != [str(v) for v in sys.version_info[:2]]:
    raise SystemExit("--resume venv version must match the selected Python 3.10/3.11 interpreter")
# Interpreter symlinks to the base Python are normal; writable directory symlinks
# must stay in the venv, otherwise pip could modify an external installation.
for directory, names, files in os.walk(root, followlinks=False):
    for name in names:
        path = Path(directory) / name
        if path.is_symlink():
            try:
                path.resolve().relative_to(root)
            except ValueError:
                raise SystemExit(f"--resume refuses external venv directory symlink: {path}")
PY
    "$venv/bin/python" -B - "$venv" <<'PY'
from pathlib import Path
import sys
import sysconfig

root = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != root or sys.prefix == sys.base_prefix:
    raise SystemExit("--resume interpreter does not belong to the requested isolated venv")
if not (3, 10) <= sys.version_info[:2] < (3, 12):
    raise SystemExit("--resume requires Python 3.10 or 3.11")
cfg = dict(line.split("=", 1) for line in (root / "pyvenv.cfg").read_text().splitlines() if "=" in line)
version = next((value.strip() for key, value in cfg.items() if key.strip().lower() == "version"), "")
if version.split(".")[:2] != [str(v) for v in sys.version_info[:2]]:
    raise SystemExit("--resume actual interpreter version differs from pyvenv.cfg")
for key in ("purelib", "platlib", "scripts", "data"):
    try:
        Path(sysconfig.get_path(key)).resolve().relative_to(root)
    except ValueError:
        raise SystemExit(f"--resume installation scheme escapes the venv: {key}")
PY
    echo "Resuming installation in the existing isolated venv: $venv"
else
    if [[ -d "$venv" && -n "$(ls -A "$venv")" ]]; then
        echo "Refusing to replace a non-empty environment: $venv. Use --resume only for an interrupted isolated installation." >&2
        exit 2
    fi
    "$py" -B -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,12), "Use Python 3.10 or 3.11"'
    "$py" -m venv "$venv"
fi
"$venv/bin/python" -m pip install 'pip>=24'
"$venv/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "${RAP_WAM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
"$venv/bin/python" -m pip install -r "$DIR/requirements.txt" -e "$XPL_ROOT"
echo "Policy environment: $venv"
