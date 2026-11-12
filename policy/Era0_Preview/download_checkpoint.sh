#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
resolve_python "${POLICY_ENV:-${POLICY_DIR}/.venv}"
"${PYTHON_BIN}" - "${POLICY_DIR}" "$@" <<'PYCODE'
import hashlib
import json
from pathlib import Path
import sys

policy = Path(sys.argv[1])
args = sys.argv[2:]
dry_run = '--dry-run' in args
args = [arg for arg in args if arg != '--dry-run']
if '--help' in args or '-h' in args:
    print('Usage: bash download_checkpoint.sh [DESTINATION] [--dry-run]\n'
          '   or: bash download_checkpoint.sh HF_REPO_ID DESTINATION [REVISION] [--dry-run]')
    raise SystemExit(0)
if len(args) > 3:
    raise SystemExit('Expected [DESTINATION] or HF_REPO_ID DESTINATION [REVISION]')
release = json.loads((policy/'checkpoint_release.json').read_text())
if release.get('release_state') == 'local_prepared' and len(args) < 2:
    raise SystemExit('This revision is prepared locally and is not published. Use the local directory from checkpoint_release.json. Explicit HF_REPO_ID DESTINATION [REVISION] is required for remote assets; the earlier public model requires its earlier adapter defaults.')
repo = args[0] if len(args) >= 2 else release['repo_id']
destination = Path(args[1] if len(args) >= 2 else args[0] if args else policy/'checkpoints/era0-preview').expanduser().resolve()
revision = args[2] if len(args) == 3 else release['revision'] if repo == release['repo_id'] else None
print(json.dumps(dict(repo_id=repo, revision=revision, local_dir=str(destination), dry_run=dry_run), indent=2), flush=True)
if dry_run:
    raise SystemExit(0)
from huggingface_hub import snapshot_download
snapshot_download(repo_id=repo, local_dir=destination, revision=revision,
                  ignore_patterns=['*.tar.gz', '*.tgz', '*.zip'])
if release.get('release_state') != 'local_prepared' and repo == release['repo_id'] and revision == release['revision']:
    norm = destination/release['norm_filename']
    if hashlib.sha256(norm.read_bytes()).hexdigest() != release['norm_sha256']:
        raise RuntimeError('Published normalization SHA256 mismatch')
    weights = destination/release['weights_filename']
    if weights.stat().st_size != release['weights_size_bytes']:
        raise RuntimeError('Published weight size mismatch')
    print('Published checkpoint downloaded; exact normalization verified. SHA256SUMS includes full weight hash.')
PYCODE
