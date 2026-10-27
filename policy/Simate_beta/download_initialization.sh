#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash download_initialization.sh [destination]"
    echo "Downloads training initialization, not final evaluation weights."
    echo "Default: pretrained/simate-beta-init under this script's directory."
    exit 0
fi
if (( $# > 1 )); then
    echo "Usage: bash download_initialization.sh [destination]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${1:-${SCRIPT_DIR}/pretrained/simate-beta-init}"

"${PYTHON:-python}" - "${DESTINATION}" <<'PYTHON'
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO_ID = "SIPAILab/simate-beta-init"
REVISION = "c924b68d04828221ed1f11e2abc7fe8e7174c24d"
REQUIRED = {"model.safetensors", "tokenizer.model", "state_action_statistics.json"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


destination = Path(sys.argv[1]).expanduser().resolve()
manifest = Path(hf_hub_download(REPO_ID, "SHA256SUMS", revision=REVISION))
checksums = {}
for line in manifest.read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    name = name.lstrip("*")
    if Path(name).name != name or name in (".", "..") or len(digest) != 64:
        raise SystemExit("Invalid initialization checksum manifest.")
    checksums[name] = digest
if not REQUIRED.issubset(checksums):
    raise SystemExit("Initialization manifest is missing required assets.")

verified = {}
for name, expected in checksums.items():
    target = destination / name
    if target.is_file() and sha256(target) == expected:
        verified[name] = target
        continue
    downloaded = Path(hf_hub_download(REPO_ID, name, revision=REVISION))
    if sha256(downloaded) != expected:
        raise SystemExit(f"Checksum mismatch: {name}; existing files were not replaced.")
    verified[name] = downloaded

with safe_open(str(verified["model.safetensors"]), framework="pt", device="cpu") as handle:
    if not any(key.startswith("backbone.history_compressor.") for key in handle.keys()):
        raise SystemExit("Initialization has no AE encoder; existing files were not replaced.")

verified["SHA256SUMS"] = manifest
destination.mkdir(parents=True, exist_ok=True)
for name, source in verified.items():
    target = destination / name
    if source.resolve() == target.resolve():
        continue
    with tempfile.TemporaryDirectory(dir=destination, prefix=".download-") as temporary:
        staged = Path(temporary) / name
        shutil.copyfile(source, staged)
        staged.replace(target)
print(f"Verified training initialization: {destination}")
print("Train the policy before reporting evaluation results.")
PYTHON
