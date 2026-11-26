#!/usr/bin/env python3
"""Verify or fetch the fixed RSIWAM base assets using only Python's standard library.

Default: dry-run, no network and no writes. Use --download explicitly to install.
The two adaptation weights are verified in the bundle and remain there.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "Haodong082399/VPP2"
REVISION = "04824ddb3aae11f13db50984ed7d540c3b77dd9f"
BASE_IDENTITY = "ae66dbfc16eb3c75fa89111501412ecb14cd3ab62ce02ff9fc8a457bb3b19aba"
MANIFEST_SHA256 = "955de560be7d371e2df31434c754c8332363d68525856a1da11db56f462c43da"
BUNDLED_METADATA = {
    "joint2b_s100000/manifest.json": "evidence/manifest.json",
    "joint2b_s100000/dataset_stats.json": "evidence/dataset_stats.json",
}
ADAPTERS = {
    "weights/parent/adapter_final.pt": "861529eb89e4279e4490d6156bf04a335eee410f6c02842d0ac3db2dc90431d3",
    "weights/step000400/adapter.pt": "eec21ec2834ef95bffdac7b3df0b48092a215aa67e7bf6952ec1643ab511b361",
}
CHUNK_BYTES = 1024 * 1024
MAX_UNSIZED_BYTES = 32 * 1024 * 1024


class AssetError(Exception):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, expected, size=None):
    if not path.is_file():
        raise AssetError(f"Expected a regular readable file: {path}")
    if size is not None and path.stat().st_size != size:
        raise AssetError(f"Size mismatch; file left unchanged: {path}")
    actual = sha256(path)
    if actual != expected:
        raise AssetError(f"SHA256 mismatch; file left unchanged: {path} (got {actual})")


def load_manifest(bundle_root):
    manifest_path = bundle_root / "base_weights.json"
    verify(manifest_path, MANIFEST_SHA256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION
            or manifest.get("base_identity") != BASE_IDENTITY):
        raise AssetError("base_weights.json does not identify the fixed RSIWAM base")
    entries = manifest.get("files", [])
    if len(entries) != 11:
        raise AssetError("Expected exactly 11 fixed base-asset entries")
    seen = set()
    for item in entries:
        rel = item.get("relative_path", "")
        parts = PurePosixPath(rel)
        if not rel or parts.is_absolute() or ".." in parts.parts or "\\" in rel or str(parts) != rel or rel in seen:
            raise AssetError(f"Unsafe or duplicate relative path: {rel!r}")
        seen.add(rel)
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise AssetError(f"Missing SHA256: {rel}")
        expected_url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/checkpoints/{rel}"
        if item.get("url") != expected_url:
            raise AssetError(f"URL does not match the public pinned repository: {rel}")
        if "bytes" in item and (type(item["bytes"]) is not int or item["bytes"] <= 0):
            raise AssetError(f"Invalid expected byte count: {rel}")
    return entries


def destination_path(destination, relative):
    path = destination / relative
    # Resolving the parent catches existing symlink escapes; the leaf is never replaced.
    if not path.parent.resolve().is_relative_to(destination.resolve()):
        raise AssetError(f"Destination path escapes the base directory: {path}")
    return path


def local_source(item, bundle_root, cache_dirs):
    rel = item["relative_path"]
    candidates = []
    if rel in BUNDLED_METADATA:
        candidates.append(bundle_root / BUNDLED_METADATA[rel])
    candidates.extend(root / rel for root in cache_dirs)
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            verify(candidate, item["sha256"], item.get("bytes"))
            return candidate
    return None


def atomic_install(stream, target, expected, size=None):
    """Hash while writing, then install atomically without replacing any existing path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    temporary = Path(name)
    digest, count = hashlib.sha256(), 0
    limit = size if size is not None else MAX_UNSIZED_BYTES
    try:
        with os.fdopen(fd, "wb") as output:
            while True:
                chunk = stream.read(min(CHUNK_BYTES, limit - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                if count > limit:
                    raise AssetError(f"Body exceeds expected/allowed byte count: {target.name}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size is not None and count != size:
            raise AssetError(f"Incomplete body for {target.name}: expected {size}, got {count}")
        if digest.hexdigest() != expected:
            raise AssetError(f"Downloaded/copied SHA256 mismatch: {target.name}")
        try:
            # link() fails if the destination already exists, including a concurrent writer.
            os.link(temporary, target)
        except FileExistsError:
            verify(target, expected, size)
            return "already-present-verified"
        return "installed-verified"
    finally:
        temporary.unlink(missing_ok=True)


def process_asset(item, bundle_root, destination, cache_dirs, *, install=False, offline=False, timeout=30):
    rel = item["relative_path"]
    target = destination_path(destination, rel)
    if target.exists() or target.is_symlink():
        verify(target, item["sha256"], item.get("bytes"))
        return "SKIP verified", rel
    source = local_source(item, bundle_root, cache_dirs)
    if not install:
        return ("PLAN copy verified local" if source else "PLAN public download"), rel
    if source is not None:
        with source.open("rb") as handle:
            status = atomic_install(handle, target, item["sha256"], item.get("bytes"))
        return "LOCAL " + status, rel
    if offline:
        raise AssetError(f"Missing verified local source in offline mode: {rel}")
    request = urllib.request.Request(item["url"], headers={
        "User-Agent": "RSIWAM-fixed-checkpoints/1", "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise AssetError(f"Unexpected HTTP {response.status}: {rel}")
            if urllib.parse.urlsplit(response.url).scheme != "https":
                raise AssetError(f"Refusing non-HTTPS redirect: {rel}")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                limit = item.get("bytes", MAX_UNSIZED_BYTES)
                if int(declared) > limit or ("bytes" in item and int(declared) != item["bytes"]):
                    raise AssetError(f"Unexpected Content-Length: {rel}")
            status = atomic_install(response, target, item["sha256"], item.get("bytes"))
        return "HTTP " + status, rel
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise AssetError(f"Public source returned HTTP {code}: {rel}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AssetError(f"Public source request failed ({type(exc).__name__}): {rel}") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parent / "bundle",
                        help="Directory containing base_weights.json, evidence/ and weights/ (default: script sibling bundle/)")
    parser.add_argument("--destination", type=Path, required=True,
                        help="Base root to contain joint2b_s100000/ and Wan2.1-I2V-14B-480P/")
    parser.add_argument("--cache-dir", type=Path, action="append", default=[],
                        help="Existing base root with the same relative layout; repeat as needed")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Install verified local copies and download missing public files")
    mode.add_argument("--dry-run", action="store_true", help="Explicit default: verify and print the plan, without network or writes")
    parser.add_argument("--offline", action="store_true", help="With --download, permit local copies only")
    parser.add_argument("--timeout", type=float, default=30, help="Network socket timeout in seconds")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        entries = load_manifest(args.bundle_root)
        for rel, expected in ADAPTERS.items():
            verify(args.bundle_root / rel, expected)
            print(f"BUNDLE verified {rel}", flush=True)
        print("MODE " + ("install offline" if args.offline else "install") if args.download else "MODE dry-run; no network, no writes", flush=True)
        for item in entries:
            status, rel = process_asset(item, args.bundle_root, args.destination, args.cache_dir,
                                        install=args.download, offline=args.offline, timeout=args.timeout)
            print(f"{status}: {rel}", flush=True)
    except (AssetError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
