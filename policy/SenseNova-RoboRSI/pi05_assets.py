"""Download and verify the exact RoboDojo step-59999 pi0.5 checkpoint.

The caller supplies the checkpoint directory (normally from XPolicyLab's shared
resolver). This module downloads public, individually hashed Orbax files; it does
not load a model, extract archives, read credentials, or use author workspaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINT_MANIFEST_SHA256 = (
    "f6e8346dc14a043b2e53396ea4ed0a953092ca555beea8c5812fcc49f121e255"
)
REPOSITORY = "RoboDojo-Benchmark/RoboDojo"
REVISION = "35efbc7dedfdbeeb6e95fb749bd885d73d483e41"
CHECKPOINT_PREFIX = "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"
CONFIG_NAME = "pi05_base_aloha_full_sim_arx-x5_seed_0"


class Pi05CheckpointError(ValueError):
    """A pinned checkpoint is unavailable or failed integrity verification."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise Pi05CheckpointError("pi0.5 checkpoint needs a relative file path")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or any(part in (".", "..") for part in path.parts)
        or path.as_posix() != value
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or "train_state" in path.parts
    ):
        raise Pi05CheckpointError("Unsafe pi0.5 checkpoint file path")
    return value


def load_pi05_metadata(policy_dir: Path = POLICY_DIR) -> dict:
    """Authenticate the committed manifest before touching a cache or network."""
    path = policy_dir / "pi05/checkpoint.json"
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256(path) != CHECKPOINT_MANIFEST_SHA256
    ):
        raise Pi05CheckpointError("Committed pi0.5 checkpoint manifest changed")
    document = json.loads(path.read_text())
    fixed = {
        "schema_version": 1,
        "repository_type": "dataset",
        "repository": REPOSITORY,
        "revision": REVISION,
        "checkpoint_prefix": CHECKPOINT_PREFIX,
        "config_name": CONFIG_NAME,
        "format": "orbax",
        "optimizer_state_included": False,
    }
    if any(document.get(k) != value for k, value in fixed.items()):
        raise Pi05CheckpointError("Unsupported pi0.5 checkpoint identity")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise Pi05CheckpointError("pi0.5 checkpoint manifest has no files")
    paths = []
    for record in files:
        if not isinstance(record, dict):
            raise Pi05CheckpointError("Invalid pi0.5 checkpoint file record")
        paths.append(_relative_path(record.get("path")))
        if (
            type(record.get("size")) is not int
            or record["size"] <= 0
            or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise Pi05CheckpointError("Invalid pi0.5 checkpoint size or SHA-256")
    if (
        len(paths) != len(set(paths))
        or type(document.get("inference_files")) is not int
        or len(files) != document["inference_files"]
        or type(document.get("inference_bytes")) is not int
        or sum(record["size"] for record in files) != document["inference_bytes"]
    ):
        raise Pi05CheckpointError("Incomplete pi0.5 checkpoint file inventory")
    return document


def _root(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise Pi05CheckpointError("pi0.5 checkpoint directory must not be a symlink")
    if path.exists() and not path.is_dir():
        raise Pi05CheckpointError("pi0.5 checkpoint directory is not a directory")
    return path.resolve()


def _target(root: Path, relative: str) -> Path:
    target = root
    for component in PurePosixPath(_relative_path(relative)).parts:
        target = target / component
        if target.is_symlink():
            raise Pi05CheckpointError(
                "pi0.5 checkpoint paths must not contain symlinks"
            )
    return target


def _verified(path: Path, record: dict) -> bool:
    if path.is_symlink():
        raise Pi05CheckpointError("pi0.5 checkpoint file must not be a symlink")
    if not path.exists():
        return False
    if (
        not path.is_file()
        or path.stat().st_size != record["size"]
        or _sha256(path) != record["sha256"]
    ):
        raise Pi05CheckpointError(
            f"pi0.5 checkpoint identity mismatch: {record['path']}"
        )
    return True


def _receipt(root: Path, metadata: dict) -> dict:
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "checkpoint": str(root),
        "config_name": CONFIG_NAME,
        "verified_files": metadata["inference_files"],
        "verified_bytes": metadata["inference_bytes"],
        "verification": "every inference file size and SHA-256",
        "optimizer_state_downloaded": False,
        "model_load_verified": False,
        "inference_verified": False,
    }


def validate_pi05_checkpoint(
    checkpoint_dir: Path, *, policy_dir: Path = POLICY_DIR
) -> dict:
    """Verify every pinned inference file, including explicit offline caches."""
    metadata = load_pi05_metadata(policy_dir)
    root = _root(checkpoint_dir)
    for record in metadata["files"]:
        if not _verified(_target(root, record["path"]), record):
            raise Pi05CheckpointError(
                f"pi0.5 checkpoint file is missing: {record['path']}; run bash install.sh"
            )
    return _receipt(root, metadata)


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise Pi05CheckpointError("pi0.5 download requires HTTPS redirects")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(record: dict, temporary: Path) -> None:
    """Resume public downloads while preserving the caller's proxy and CA setup."""
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise Pi05CheckpointError("Invalid pi0.5 partial download path")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > record["size"]:
        raise Pi05CheckpointError("pi0.5 partial download exceeds pinned file size")
    if offset < record["size"]:
        url = (
            f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
            f"{CHECKPOINT_PREFIX}/{record['path']}"
        )
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = Request(url, headers=headers)
        try:
            with build_opener(_HTTPSRedirect()).open(request, timeout=60) as response:
                if offset and response.status == 206:
                    wanted = f"bytes {offset}-{record['size'] - 1}/{record['size']}"
                    if response.headers.get("Content-Range") != wanted:
                        raise Pi05CheckpointError(
                            "pi0.5 server returned an incorrect download range"
                        )
                elif response.status == 200:
                    offset = 0
                else:
                    raise Pi05CheckpointError(
                        "pi0.5 server returned an unexpected HTTP status"
                    )
                remaining = record["size"] - offset
                with temporary.open("ab" if offset else "wb") as stream:
                    while remaining:
                        block = response.read(min(4 * 1024 * 1024, remaining))
                        if not block:
                            raise Pi05CheckpointError(
                                "Truncated pi0.5 checkpoint download"
                            )
                        stream.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise Pi05CheckpointError(
                            "pi0.5 download exceeds pinned file size"
                        )
        except HTTPError as error:
            raise Pi05CheckpointError(
                f"pi0.5 checkpoint download failed (HTTP {error.code}): {record['path']}"
            ) from None
        except (OSError, URLError) as error:
            # Exception strings may contain signed CDN URLs; never log them.
            raise Pi05CheckpointError(
                f"pi0.5 checkpoint download failed ({type(error).__name__}): {record['path']}"
            ) from None
    try:
        _verified(temporary, record)
    except Pi05CheckpointError:
        temporary.unlink(missing_ok=True)
        raise


def install_pi05_checkpoint(
    checkpoint_dir: Path,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    workers: int = 4,
    policy_dir: Path = POLICY_DIR,
) -> dict:
    """Install exact files into a caller-resolved directory without path fallbacks.

    An explicit cache has the same layout as the checkpoint root. Existing cache
    files are verified before copying. Missing files download directly into the
    destination, so the default installation needs no second checkpoint copy.
    Changed destination/cache files are preserved and rejected.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise Pi05CheckpointError("pi0.5 download workers must be between 1 and 8")
    metadata = load_pi05_metadata(policy_dir)
    root = _root(checkpoint_dir)
    cache = _root(cache_dir) if cache_dir is not None else None
    for record in metadata["files"]:
        _verified(_target(root, record["path"]), record)
    if cache is not None and cache != root:
        # Authenticate every present cache file before using any of its data.
        for record in metadata["files"]:
            _verified(_target(cache, record["path"]), record)
    root.mkdir(parents=True, exist_ok=True)

    def install(record):
        target = _target(root, record["path"])
        if _verified(target, record):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        if temporary.is_symlink():
            raise Pi05CheckpointError("pi0.5 partial download must not be a symlink")
        cached = _target(cache, record["path"]) if cache is not None else None
        if cached is not None and cached != target and _verified(cached, record):
            shutil.copyfile(cached, temporary)
        elif offline:
            raise Pi05CheckpointError(
                f"pi0.5 checkpoint unavailable offline: {record['path']}"
            )
        else:
            _download(record, temporary)
        if not _verified(temporary, record):
            raise Pi05CheckpointError("pi0.5 checkpoint staging file is missing")
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(install, metadata["files"]))
    return _receipt(root, metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Verify only; never download"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.check:
            result = validate_pi05_checkpoint(args.checkpoint_dir)
        else:
            result = install_pi05_checkpoint(
                args.checkpoint_dir,
                cache_dir=args.cache_dir,
                offline=args.offline,
                workers=args.workers,
            )
    except (OSError, ValueError) as error:
        print(f"pi0.5 checkpoint preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
