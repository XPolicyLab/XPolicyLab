"""Prepare the reference PaliGemma tokenizer under explicit OPENPI_DATA_HOME."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

try:
    from .pi05_assets import (
        Pi05CheckpointError,
        _HTTPSRedirect,
        _root,
        _sha256,
        _target,
    )
except ImportError:
    from pi05_assets import Pi05CheckpointError, _HTTPSRedirect, _root, _sha256, _target

POLICY_DIR = Path(__file__).resolve().parent
TOKENIZER_MANIFEST_SHA256 = (
    "90ed7bfc4364e7eeb87797072c22c7d715b923f20c0bde780cfc81c2283cb1d0"
)
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"


def load_pi05_tokenizer_metadata(policy_dir: Path = POLICY_DIR) -> dict:
    manifest = policy_dir / "pi05/tokenizer.json"
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or _sha256(manifest) != TOKENIZER_MANIFEST_SHA256
    ):
        raise Pi05CheckpointError("Committed pi0.5 tokenizer manifest changed")
    document = json.loads(manifest.read_text())
    fixed = {
        "schema_version": 1,
        "upstream_uri": "gs://big_vision/paligemma_tokenizer.model",
        "url": TOKENIZER_URL,
        "cache_relative_path": "big_vision/paligemma_tokenizer.model",
        "size": 4_264_023,
        "sha256": TOKENIZER_SHA256,
    }
    if any(document.get(k) != value for k, value in fixed.items()):
        raise Pi05CheckpointError("Unsupported pi0.5 tokenizer identity")
    return document


def _verified_tokenizer(path: Path, metadata: dict) -> bool:
    if path.is_symlink():
        raise Pi05CheckpointError("pi0.5 tokenizer must not be a symlink")
    if not path.exists():
        return False
    if (
        not path.is_file()
        or path.stat().st_size != metadata["size"]
        or _sha256(path) != metadata["sha256"]
    ):
        raise Pi05CheckpointError("pi0.5 tokenizer size or SHA-256 mismatch")
    return True


def validate_pi05_tokenizer(data_home: Path, *, policy_dir: Path = POLICY_DIR) -> dict:
    """Check the exact cache location used by unmodified OpenPI download code."""
    metadata = load_pi05_tokenizer_metadata(policy_dir)
    root = _root(data_home)
    target = _target(root, metadata["cache_relative_path"])
    if not _verified_tokenizer(target, metadata):
        raise Pi05CheckpointError("pi0.5 tokenizer is missing; run bash install.sh")
    return {
        "path": str(target),
        "OPENPI_DATA_HOME": str(root),
        "bytes": metadata["size"],
        "sha256": metadata["sha256"],
        "verification": "full tokenizer size and SHA-256",
    }


def install_pi05_tokenizer(
    data_home: Path,
    *,
    cache_file: Path | None = None,
    offline: bool = False,
    policy_dir: Path = POLICY_DIR,
) -> dict:
    """Download the public object or copy an explicitly supplied, verified cache."""
    metadata = load_pi05_tokenizer_metadata(policy_dir)
    root = _root(data_home)
    target = _target(root, metadata["cache_relative_path"])
    cached = Path(cache_file).expanduser() if cache_file is not None else None
    if cached is not None and not _verified_tokenizer(cached, metadata):
        raise Pi05CheckpointError("Explicit pi0.5 tokenizer cache file is missing")
    if _verified_tokenizer(target, metadata):
        return validate_pi05_tokenizer(root, policy_dir=policy_dir)
    if offline and cached is None:
        raise Pi05CheckpointError("pi0.5 tokenizer is unavailable offline")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".pi05-tokenizer-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            if cached is not None:
                with cached.open("rb") as source:
                    shutil.copyfileobj(source, stream)
            else:
                try:
                    request = Request(metadata["url"])
                    with build_opener(_HTTPSRedirect()).open(
                        request, timeout=60
                    ) as response:
                        if response.status != 200:
                            raise Pi05CheckpointError(
                                "Unexpected pi0.5 tokenizer HTTP status"
                            )
                        remaining = metadata["size"]
                        while remaining:
                            block = response.read(min(1024 * 1024, remaining))
                            if not block:
                                raise Pi05CheckpointError(
                                    "Truncated pi0.5 tokenizer download"
                                )
                            stream.write(block)
                            remaining -= len(block)
                        if response.read(1):
                            raise Pi05CheckpointError(
                                "pi0.5 tokenizer download exceeds pinned size"
                            )
                except HTTPError as error:
                    raise Pi05CheckpointError(
                        f"pi0.5 tokenizer download failed (HTTP {error.code})"
                    ) from None
                except (OSError, URLError) as error:
                    raise Pi05CheckpointError(
                        f"pi0.5 tokenizer download failed ({type(error).__name__})"
                    ) from None
            stream.flush()
            if not _verified_tokenizer(temporary, metadata):
                raise Pi05CheckpointError("pi0.5 tokenizer staging file is missing")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return validate_pi05_tokenizer(root, policy_dir=policy_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-home",
        type=Path,
        required=True,
        help="OPENPI_DATA_HOME used by the VLA service",
    )
    parser.add_argument("--cache-file", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            receipt = validate_pi05_tokenizer(args.data_home)
        else:
            receipt = install_pi05_tokenizer(
                args.data_home, cache_file=args.cache_file, offline=args.offline
            )
    except (OSError, ValueError) as error:
        print(f"pi0.5 tokenizer preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
