#!/usr/bin/env python3
"""Download or verify a CR-WM asset manifest using only the Python standard library.

No checkpoint is deserialized and no archive is extracted. Frozen-model and
dataset-metadata URLs and the public CR-WM release URL are pinned.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import BinaryIO
from urllib.error import URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_MANIFEST = Path(__file__).resolve().with_name("weights_manifest.json")
CHUNK_BYTES = 8 * 1024 * 1024
URL_PLACEHOLDERS = frozenset({
    "__CRWM_BF16_INFERENCE_URL__",
    "__CRWM_FP32_INITIALIZATION_URL__",
})


class AssetError(Exception):
    """A manifest, source, or local asset failed validation."""


@dataclass(frozen=True)
class Asset:
    path: str
    bytes: int
    sha256: str
    url: str | None


def relative_asset_path(value: object) -> str:
    """Accept portable, normalized POSIX paths inside the selected asset root."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise AssetError(f"Invalid asset path: {value!r}")
    if "\x00" in value:
        raise AssetError("NUL is not allowed in an asset path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise AssetError(f"Asset path must be a normalized relative path: {value!r}")
    return path.as_posix()


def http_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
    except ValueError as exc:
        raise AssetError(f"Invalid download URL: {value!r}") from exc
    if parsed.scheme.lower() not in ("http", "https") or not host:
        raise AssetError("Download URLs must use http:// or https://; use --source-dir for local files")
    if parsed.username is not None or parsed.password is not None:
        raise AssetError("Credentials embedded in download URLs are not supported")
    if parsed.fragment:
        raise AssetError("Download URLs must not contain fragments")
    return value


def load_manifest(path: Path) -> list[Asset]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssetError(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AssetError("The manifest must be a JSON object")
    if manifest.get("pending") is True:
        raise AssetError(
            f"{path.name} is pending release details. Replace its URL placeholder, "
            "fill verified bytes/SHA256, then set pending=false."
        )
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise AssetError("The manifest must contain a nonempty 'files' list")
    assets: list[Asset] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AssetError(f"Manifest files[{index}] must be an object")
        relative = relative_asset_path(row.get("path"))
        if relative in seen:
            raise AssetError(f"Duplicate manifest path: {relative}")
        seen.add(relative)
        size = row.get("bytes", row.get("size"))
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise AssetError(f"Missing or invalid byte size for {relative}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise AssetError(f"Missing or invalid SHA256 for {relative}")
        url = row.get("url")
        if isinstance(url, str) and url in URL_PLACEHOLDERS:
            url = None
        if url is not None:
            if not isinstance(url, str) or not url:
                raise AssetError(f"URL for {relative} must be null or a nonempty string")
            http_url(url)
        assets.append(Asset(relative, size, digest.lower(), url))
    return assets


def safe_path(root: Path, relative: str) -> Path:
    """Reject every symlink below root, including links pointing back inside it."""
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        if current.is_symlink():
            raise AssetError(f"Symlinks are not permitted in asset paths: {current}")
    try:
        current.resolve().relative_to(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise AssetError(f"Asset path escapes the selected root: {relative}") from exc
    return current


def check_file(path: Path, asset: Asset) -> None:
    if not path.is_file():
        raise AssetError(f"Missing regular file: {path}")
    size = path.stat().st_size
    if size != asset.bytes:
        raise AssetError(f"Size mismatch for {asset.path}: expected {asset.bytes}, got {size}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    if digest.hexdigest() != asset.sha256:
        raise AssetError(f"SHA256 mismatch for {asset.path}")


class HttpOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def asset_url(asset: Asset, base_url: str | None) -> str:
    if asset.url:
        return asset.url
    if base_url:
        return http_url(urljoin(base_url, quote(asset.path, safe="/")))
    raise AssetError(
        f"Download URL is not configured for {asset.path}. Replace its URL placeholder "
        "in the manifest with a direct HTTP(S) URL. Alternatively supply "
        "--base-url, or --source-dir; use --verify-only for existing files."
    )


def copy_checked(source: BinaryIO, target: BinaryIO, asset: Asset) -> None:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(CHUNK_BYTES), b""):
        size += len(block)
        if size > asset.bytes:
            raise AssetError(f"Source is larger than the manifest size for {asset.path}")
        digest.update(block)
        target.write(block)
    if size != asset.bytes:
        raise AssetError(f"Size mismatch for {asset.path}: expected {asset.bytes}, got {size}")
    if digest.hexdigest() != asset.sha256:
        raise AssetError(f"SHA256 mismatch for {asset.path}")


def write_asset(
    asset: Asset,
    root: Path,
    *,
    source_dir: Path | None,
    base_url: str | None,
    replace: bool,
    timeout: float,
) -> None:
    destination = safe_path(root, asset.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_path(root, asset.path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".crwm-download-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            if source_dir is not None:
                local_source = safe_path(source_dir, asset.path)
                if not local_source.is_file():
                    raise AssetError(f"Local source is not a regular file: {local_source}")
                with local_source.open("rb") as source:
                    copy_checked(source, target, asset)
            else:
                request = Request(asset_url(asset, base_url), headers={"User-Agent": "CR-WM-assets/1"})
                with build_opener(HttpOnlyRedirectHandler()).open(request, timeout=timeout) as source:
                    copy_checked(source, target, asset)
            target.flush()
            os.fsync(target.fileno())
        temporary.chmod(0o644)
        safe_path(root, asset.path)
        if replace:
            if destination.exists() and not destination.is_file():
                raise AssetError(f"Refusing to replace a non-file destination: {destination}")
            os.replace(temporary, destination)
        else:
            # Both files are on the same filesystem. link() atomically refuses an
            # existing destination rather than clobbering a file created meanwhile.
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise AssetError(f"Destination appeared during download; not overwritten: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    assets = load_manifest(args.manifest)
    root = args.output_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve() if args.source_dir is not None else None
    if source_dir is not None and not source_dir.is_dir():
        raise AssetError(f"Local source directory does not exist: {source_dir}")
    base_url = args.base_url
    if base_url:
        http_url(base_url)
        if urlsplit(base_url).query:
            raise AssetError("--base-url must not contain a query; use per-file URLs for signed downloads")
        base_url = base_url.rstrip("/") + "/"
    pending: list[Asset] = []
    for asset in assets:
        destination = safe_path(root, asset.path)
        if destination.exists():
            try:
                check_file(destination, asset)
            except AssetError as exc:
                if args.verify_only or not args.replace:
                    raise AssetError(f"{exc}. Existing files were not changed; use --replace to replace a bad file.") from exc
                if not destination.is_file():
                    raise AssetError(f"Refusing to replace a non-file destination: {destination}")
                pending.append(asset)
            else:
                print(f"[verified] {asset.path}")
        elif args.verify_only:
            raise AssetError(f"Missing file: {destination}; --verify-only never downloads")
        else:
            pending.append(asset)
        if source_dir is not None:
            safe_path(source_dir, asset.path)
    # Validate all required sources before creating directories or copying data.
    for asset in pending:
        if source_dir is None:
            asset_url(asset, base_url)
        elif not safe_path(source_dir, asset.path).is_file():
            raise AssetError(f"Missing local source: {asset.path}")
    if pending:
        root.mkdir(parents=True, exist_ok=True)
    for asset in pending:
        write_asset(asset, root, source_dir=source_dir, base_url=base_url,
                    replace=args.replace, timeout=args.timeout)
        print(f"[downloaded] {asset.path}")
    print(f"Verified {len(assets)} files ({sum(asset.bytes for asset in assets)} bytes); "
          f"downloaded/copied {len(pending)}.")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Download and SHA256-verify CR-WM assets. No archives are extracted or models loaded.",
        epilog=("Examples:\n"
                "  python download_checkpoint.py --output-dir /path/to/release --base-url https://host/release/\n"
                "  python download_checkpoint.py --output-dir /path/to/release --verify-only\n"
                "  python download_checkpoint.py --output-dir /path/to/copy --source-dir /path/to/assets\n"
                "  python download_checkpoint.py --manifest /path/to/training-release.json --output-dir /path/to/training\n"
                "Manifest: {\"files\": [{\"path\": \"weights/model.pt\", \"bytes\": 123, \"sha256\": \"<64 hex>\", \"url\": null}]}.\n"
                "Per-file URLs take precedence over --base-url. 'size' is accepted as an alias for 'bytes'.\n"
                "The default inference manifest pins eight files, including the public CR-WM checkpoint.\n"
                "backbone_weights_manifest.json downloads the seven frozen inference assets independently.\n"
                "training_weights_manifest.json downloads the frozen training VAE/DINO.\n"
                "dataset_metadata_manifest.json downloads the original native dataset metadata."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cli.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Local JSON manifest (default: weights_manifest.json beside this script)")
    cli.add_argument("--output-dir", type=Path, required=True, help="Explicit asset root; manifest paths are relative to it")
    sources = cli.add_mutually_exclusive_group()
    sources.add_argument("--base-url", help="HTTP(S) release root for entries whose URL is null or still a placeholder")
    sources.add_argument("--source-dir", type=Path, help="Copy from an explicit local asset root instead of downloading")
    cli.add_argument("--verify-only", action="store_true", help="Verify existing local files only; never connects to the network or writes files")
    cli.add_argument("--replace", action="store_true", help="Atomically replace existing bad regular files after size and SHA256 validation")
    cli.add_argument("--timeout", type=float, default=60.0, help="HTTP socket timeout in seconds (default: 60)")
    return cli


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.timeout <= 0:
        cli.error("--timeout must be greater than zero")
    if args.verify_only and args.replace:
        cli.error("--replace cannot be combined with --verify-only")
    try:
        run(args)
    except (AssetError, OSError, URLError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
