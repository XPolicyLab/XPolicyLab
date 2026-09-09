"""Provision the pinned PaliGemma tokenizer in OpenPI's standard cache."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
TOKENIZER_PATH = Path("big_vision/paligemma_tokenizer.model")


def verify(path: Path) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != TOKENIZER_SHA256:
        raise ValueError(f"Tokenizer SHA-256 mismatch at {path}: {actual}; expected {TOKENIZER_SHA256}")


def prepare(cache_dir: Path, source_file: Path | None = None, check: bool = False) -> Path:
    destination = cache_dir.expanduser().resolve() / TOKENIZER_PATH
    if destination.exists():
        verify(destination)
        print(f"[KinRT] Verified tokenizer: {destination}")
        return destination
    if check:
        raise FileNotFoundError(f"Tokenizer is not cached: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="paligemma-", suffix=".partial", dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            if source_file is not None:
                with source_file.open("rb") as source:
                    shutil.copyfileobj(source, target)
            else:
                with urllib.request.urlopen(TOKENIZER_URL, timeout=120) as source:
                    shutil.copyfileobj(source, target)
        verify(temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"[KinRT] Prepared tokenizer: {destination} (SHA-256 {TOKENIZER_SHA256})")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")))
    parser.add_argument("--source-file", type=Path, help="Use a local copy with the same required SHA-256")
    parser.add_argument("--check", action="store_true", help="Verify an existing cache without downloading")
    args = parser.parse_args()
    prepare(args.cache_dir, args.source_file, args.check)
