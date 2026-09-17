#!/bin/bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash download_checkpoint.sh [destination]"
    echo "Default: checkpoints/sipai-robodojo-eval under this script's directory."
    exit 0
fi
if (( $# > 1 )); then
    echo "Usage: bash download_checkpoint.sh [destination]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${1:-${SCRIPT_DIR}/checkpoints/sipai-robodojo-eval}"

python - "${DESTINATION}" <<'PY'
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

REPO_ID = "SIPAILab/sipai-robodojo-eval"
TOKENIZER_REPO = "google/paligemma-3b-pt-224"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo, name, destination, expected=None):
    target = destination / name
    if expected and target.is_file() and sha256(target) == expected:
        print(f"Reusing verified file: {target}", flush=True)
        return
    metadata = get_hf_file_metadata(hf_hub_url(repo, name, revision="main"))
    checksum = expected or metadata.etag
    if not checksum or len(checksum) != 64:
        raise SystemExit(f"Missing SHA-256 metadata for {repo}/{name}.")
    if expected and metadata.etag != expected:
        raise SystemExit("The upstream tokenizer differs from the tokenizer supported by this adapter.")
    if target.is_file() and sha256(target) == checksum:
        print(f"Reusing current file: {target}", flush=True)
        return
    # Resolve main once, then use that same snapshot even if main changes mid-download.
    source = Path(hf_hub_download(repo_id=repo, filename=name, revision=metadata.commit_hash))
    if sha256(source) != checksum:
        raise SystemExit(f"Checksum mismatch for {name}; the existing destination was not replaced.")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix=f".{name}.") as temporary:
        staged = Path(temporary) / name
        shutil.copyfile(source, staged)
        staged.replace(target)
    print(f"Ready: {target}", flush=True)


destination = Path(sys.argv[1]).expanduser().resolve()
print(f"Downloading current SIPAI checkpoint from {REPO_ID}", flush=True)
try:
    # Resolve tokenizer access before downloading the much larger model file.
    fetch(TOKENIZER_REPO, "tokenizer.model", destination, TOKENIZER_SHA256)
    fetch(REPO_ID, "model.safetensors", destination)
except GatedRepoError:
    raise SystemExit(f"Accept the access terms at https://huggingface.co/{TOKENIZER_REPO}, then run hf auth login or set HF_TOKEN.") from None
except RepositoryNotFoundError:
    raise SystemExit("Cannot access the Hugging Face repository. If private, run hf auth login or set HF_TOKEN with read access.") from None
print(f"Checkpoint directory: {destination}")
PY
