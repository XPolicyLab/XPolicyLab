#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kind=${1:-original}
case "$kind" in original|merged|training) ;; *) echo "Use original, merged or training" >&2; exit 2 ;; esac
output=${2:-"$SCRIPT_DIR/checkpoints/worldla-100k-decay"}
python - "$kind" "$output" <<'PY'
import sys
from huggingface_hub import snapshot_download
kind, output = sys.argv[1:]
patterns = ['training/stage_routes_20k/*'] if kind == 'training' else [kind + '/*', 'README.md']
snapshot_download('xhx0327/WorldLA-100k-decay-full',
    revision='7fd1111c6fa8419a64bb6f26383b5025ffbf436d',
    allow_patterns=patterns, local_dir=output, token=False, max_workers=4)
print('Downloaded:', output)
PY
