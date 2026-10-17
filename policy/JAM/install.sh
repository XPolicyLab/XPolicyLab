#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=runtime
VENV=""
DRY_RUN=0
while (($#)); do
  case "$1" in
    --mode) MODE=${2:?}; shift 2;;
    --venv) VENV=${2:?}; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) echo 'Usage: bash install.sh --venv /path/to/new/venv [--mode runtime|training|data] [--dry-run]'; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$VENV" && "$VENV" = /* ]] || { echo '--venv must be an explicit absolute path' >&2; exit 2; }
case "$MODE" in runtime|training|data) ;; *) echo 'Unknown install mode' >&2; exit 2;; esac
PYTHON=${PYTHON:-python3.10}
run() {
  printf '%q ' "$@"; printf '\n'
  if (( ! DRY_RUN )); then "$@"; fi
}
if [[ -f "$VENV/pyvenv.cfg" ]] && grep -qi 'include-system-site-packages = true' "$VENV/pyvenv.cfg"; then
  echo 'Use a new isolated venv, without inherited site-packages.' >&2; exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  run "$PYTHON" -m venv "$VENV"
fi
run "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required for this candidate"'
run "$VENV/bin/python" -m pip install 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
run "$VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple 'torch==2.7.1+cu128' 'torchvision==0.22.1'
export DS_BUILD_OPS=0
if [[ "$MODE" == data ]]; then
  run "$VENV/bin/python" -m pip install -r "$POLICY_DIR/requirements/data.txt"
elif [[ "$MODE" == training ]]; then
  run "$VENV/bin/python" -m pip install --no-build-isolation -r "$POLICY_DIR/requirements/training.txt"
else
  run "$VENV/bin/python" -m pip install -r "$POLICY_DIR/requirements/$MODE.txt"
fi
run "$VENV/bin/python" -m pip check
echo "Environment: $VENV; activate it before serving or running an optional preparation workflow."
