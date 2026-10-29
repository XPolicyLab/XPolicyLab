#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_dir=${1:-"$SCRIPT_DIR/.venv"}
mode=${2:-runtime}
case "$mode" in runtime|train) ;; *) echo "Mode must be runtime or train" >&2; exit 2 ;; esac
test ! -e "$env_dir" || { echo "Refusing an existing environment: $env_dir" >&2; exit 1; }
python3.11 -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
source_dir=${WORLDLA_SOURCE_DIR:-"$SCRIPT_DIR/worldla-src"}
if [[ ! -d "$source_dir/.git" ]]; then
    test ! -e "$source_dir" || { echo "Refusing to overwrite $source_dir" >&2; exit 1; }
    git clone https://github.com/18643638196xhx/WorldLA.git "$source_dir"
fi
# Known public commit; adapter only depends on the stable worldla.policy API.
git -C "$source_dir" rev-parse --verify d84647cb4f412af1a9b852146073e42b4e3b76ea^{commit} >/dev/null
git -C "$source_dir" diff --quiet && git -C "$source_dir" diff --cached --quiet || {
    echo "WorldLA source has edits; use a separate checkout." >&2; exit 1;
}
git -C "$source_dir" checkout d84647cb4f412af1a9b852146073e42b4e3b76ea
if [[ "$mode" == train ]]; then req=requirements-train.txt; else req=requirements-runtime.txt; fi
"$env_dir/bin/python" -m pip install -r "$source_dir/$req"
"$env_dir/bin/python" -m pip install -e "$source_dir" --no-deps
"$env_dir/bin/python" -m pip install \
    websockets==14.1 msgpack==1.1.0 msgpack-numpy==0.4.8 h5py==3.12.1
"$env_dir/bin/python" -m pip install -e "$XPL_ROOT" --no-deps
