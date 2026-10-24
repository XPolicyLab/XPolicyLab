# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fetch fixed context packages into a verified, transactional local cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .build import _checked_absolute, _portable_path, _tree_entries
from .package import ContextPackage

_CHUNK_BYTES = 64 * 1024


def _digest(value: str, field: str, *, length: int = 64) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
    ):
        raise ValueError(
            f"{field} must be a full lowercase SHA-{256 if length == 64 else 1} hex digest"
        )


def _relative(value: str, *, allow_root: bool = False) -> PurePosixPath:
    if allow_root and value == ".":
        return PurePosixPath(value)
    return PurePosixPath(_portable_path(value, "package path"))


def _http_url(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(char) <= 32 for char in value)
    ):
        raise ValueError("HTTP package URL must be an HTTP or HTTPS URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("invalid HTTP package URL") from None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "\\" in value
    ):
        raise ValueError(
            "HTTP package URL needs an HTTP(S) host without credentials or fragment"
        )
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != (443 if scheme == "https" else 80):
        host += f":{port}"
    return urlunsplit((scheme, host, parsed.path or "/", parsed.query, ""))


@dataclass(frozen=True)
class LocalPackageSource:
    """An existing sealed package, identified by its path and manifest hash."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class HttpArchiveSource:
    """A ZIP package pinned to the SHA-256 of the complete archive bytes."""

    url: str
    archive_sha256: str
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _http_url(self.url))
        _digest(self.archive_sha256, "archive_sha256")
        if self.manifest_sha256 is not None:
            _digest(self.manifest_sha256, "manifest_sha256")


@dataclass(frozen=True)
class HfPackageSource:
    """A package subtree at a complete immutable Hugging Face commit ID."""

    repo_id: str
    revision: str
    subpath: str
    repo_type: str = "dataset"
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repo_id, str)
            or len(self.repo_id) > 193
            or re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", self.repo_id, re.ASCII) is None
            or any(
                part.startswith((".", "-")) or part.endswith((".", "-"))
                for part in self.repo_id.split("/")
            )
            or ".." in self.repo_id
            or "--" in self.repo_id
        ):
            raise ValueError("HF repo_id must name one repository")
        _digest(self.revision, "HF revision", length=40)
        _relative(self.subpath, allow_root=True)
        if self.repo_type not in {"dataset", "model", "space"}:
            raise ValueError("HF repo_type must be dataset, model, or space")
        if self.manifest_sha256 is not None:
            _digest(self.manifest_sha256, "manifest_sha256")


def _identity(
    source: LocalPackageSource | HttpArchiveSource | HfPackageSource,
) -> tuple[dict[str, Any], ContextPackage | None]:
    if isinstance(source, LocalPackageSource):
        root = _checked_absolute(source.path)
        if not root.is_dir():
            raise ValueError("local package must be a real directory")
        _tree_entries(root)
        package = ContextPackage.load(root)
        return {
            "kind": "local",
            "path": str(package.root),
            "manifest_sha256": package.manifest_sha256,
        }, package
    if isinstance(source, HttpArchiveSource):
        parsed = urlsplit(source.url)
        return {
            "kind": "http",
            "origin": urlunsplit((parsed.scheme, parsed.netloc, "", "", "")),
            "url_sha256": hashlib.sha256(source.url.encode("utf-8")).hexdigest(),
            "archive_sha256": source.archive_sha256,
            "manifest_sha256": source.manifest_sha256,
        }, None
    if isinstance(source, HfPackageSource):
        return {
            "kind": "hf",
            "repo_id": source.repo_id,
            "repo_type": source.repo_type,
            "revision": source.revision,
            "subpath": source.subpath,
            "manifest_sha256": source.manifest_sha256,
        }, None
    raise TypeError(
        "source must be LocalPackageSource, HttpArchiveSource, or HfPackageSource"
    )


def _check_manifest(package: ContextPackage, identity: dict[str, Any]) -> None:
    expected = identity.get("manifest_sha256")
    if expected is not None and package.manifest_sha256 != expected:
        raise ValueError(
            "context package manifest SHA-256 differs from the source hash"
        )


def _read_cache(entry: Path, identity: dict[str, Any]) -> ContextPackage:
    _checked_absolute(entry)
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError("context package cache entry must be a real directory")
    if {path.name for path in entry.iterdir()} != {"package", "receipt.json"}:
        raise ValueError(
            "context package cache entry is incomplete or contains unexpected files"
        )
    receipt_path = entry / "receipt.json"
    info = receipt_path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("context package receipt must be a regular unlinked file")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid context package cache receipt") from None
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"schema_version", "source", "manifest_sha256"}
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["source"] != identity
    ):
        raise ValueError(
            "context package cache receipt differs from the requested source"
        )
    _tree_entries(entry / "package")
    package = ContextPackage.load(entry / "package")
    if receipt["manifest_sha256"] != package.manifest_sha256:
        raise ValueError("context package cache manifest hash differs from its receipt")
    _check_manifest(package, identity)
    return package


@dataclass
class _ByteBudget:
    limit: int
    used: int = 0

    def add(self, count: int) -> None:
        if count > self.limit - self.used:
            raise ValueError("context package byte limit exceeded")
        self.used += count


def _copy_local(
    package: ContextPackage, destination: Path, budget: _ByteBudget
) -> None:
    destination.mkdir()
    for path, _ in _tree_entries(package.root):
        _checked_absolute(path)
        relative = path.relative_to(package.root)
        target = destination / relative
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            target.mkdir()
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as source, target.open("xb") as output:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise ValueError(
                        "local package files must remain regular and unlinked"
                    )
                while chunk := source.read(_CHUNK_BYTES):
                    budget.add(len(chunk))
                    output.write(chunk)
        else:
            raise ValueError("local package cannot contain links or special files")


def _download(
    client: Any,
    url: str,
    destination: Path,
    budget: _ByteBudget,
    *,
    unpacked: _ByteBudget | None = None,
) -> str:
    import httpx

    digest = hashlib.sha256()
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            if (
                response.headers.get("content-encoding", "identity").lower()
                != "identity"
            ):
                raise ValueError("package downloads require identity content encoding")
            with destination.open("xb") as output:
                for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                    budget.add(len(chunk))
                    if unpacked is not None:
                        unpacked.add(len(chunk))
                    output.write(chunk)
                    digest.update(chunk)
    except httpx.HTTPError as exc:
        status = (
            f", status {exc.response.status_code}"
            if isinstance(exc, httpx.HTTPStatusError)
            else ""
        )
        raise RuntimeError(
            f"context package download failed ({type(exc).__name__}{status})"
        ) from None
    return digest.hexdigest()


class _Paths:
    """Reject portable-name collisions before writing untrusted archive paths."""

    def __init__(self) -> None:
        self.explicit: set[str] = set()
        self.names: dict[str, tuple[str, bool]] = {}

    def add(self, name: str, is_dir: bool) -> PurePosixPath:
        path = _relative(name)
        if name in self.explicit:
            raise ValueError("duplicate package path")
        self.explicit.add(name)
        for index in range(1, len(path.parts) + 1):
            prefix = "/".join(path.parts[:index])
            directory = is_dir or index < len(path.parts)
            key = unicodedata.normalize("NFC", prefix).casefold()
            previous = self.names.get(key)
            if previous is not None and previous != (prefix, directory):
                raise ValueError("package contains conflicting paths")
            self.names[key] = (prefix, directory)
        if path.parts[0] not in {"manifest.json", "payload"}:
            raise ValueError("archive root needs only manifest.json and payload")
        if path.parts[0] == "manifest.json" and (is_dir or len(path.parts) != 1):
            raise ValueError("archive manifest.json must be a file at the root")
        if path.parts == ("payload",) and not is_dir:
            raise ValueError("archive payload must be a directory")
        return path


def _unpack(archive_path: Path, destination: Path, budget: _ByteBudget) -> None:
    destination.mkdir()
    paths = _Paths()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = []
            for entry in archive.infolist():
                # ZipInfo truncates names at NUL; reject the original, too.
                if (
                    entry.orig_filename != entry.filename
                    or "\x00" in entry.orig_filename
                ):
                    raise ValueError("ZIP paths cannot contain NUL")
                is_dir = entry.is_dir()
                name = entry.filename[:-1] if is_dir else entry.filename
                relative = paths.add(name, is_dir)
                mode = stat.S_IFMT(entry.external_attr >> 16)
                expected = stat.S_IFDIR if is_dir else stat.S_IFREG
                if mode not in (0, expected) or entry.flag_bits & 1:
                    raise ValueError(
                        "ZIP entries must be unencrypted regular files or directories"
                    )
                entries.append((entry, relative, is_dir))
            if sum(entry.file_size for entry, _, _ in entries) > budget.limit:
                raise ValueError("context package unpacked byte limit exceeded")
            for entry, relative, is_dir in entries:
                target = destination.joinpath(*relative.parts)
                if is_dir:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as output:
                    while chunk := source.read(_CHUNK_BYTES):
                        budget.add(len(chunk))
                        output.write(chunk)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
        raise ValueError("invalid or unsupported context package ZIP archive") from None


def _fetch_http(
    source: HttpArchiveSource,
    staging: Path,
    destination: Path,
    download: _ByteBudget,
    unpacked: _ByteBudget,
) -> None:
    import httpx

    archive_path = staging / "download.zip"
    with httpx.Client(
        follow_redirects=True, headers={"Accept-Encoding": "identity"}
    ) as client:
        actual = _download(client, source.url, archive_path, download)
    if actual != source.archive_sha256:
        raise ValueError("context package archive SHA-256 differs from the source hash")
    _unpack(archive_path, destination, unpacked)


def _fetch_hf(
    source: HfPackageSource,
    destination: Path,
    download: _ByteBudget,
    unpacked: _ByteBudget,
) -> None:
    import httpx
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    try:
        info = HfApi().repo_info(
            source.repo_id,
            revision=source.revision,
            repo_type=source.repo_type,
            files_metadata=True,
        )
        headers = build_hf_headers()
    except Exception as exc:
        raise RuntimeError(
            f"HF package metadata request failed ({type(exc).__name__})"
        ) from None
    if info.sha != source.revision:
        raise ValueError("HF returned a different revision from the pinned commit ID")
    prefix = "" if source.subpath == "." else source.subpath + "/"
    paths = _Paths()
    selected = []
    for sibling in info.siblings or ():
        filename = sibling.rfilename
        if not isinstance(filename, str) or not filename.startswith(prefix):
            continue
        relative = filename[len(prefix) :]
        if relative != "manifest.json" and not relative.startswith("payload/"):
            continue
        paths.add(relative, False)
        size = sibling.size
        if type(size) is not int or size < 0:
            raise ValueError(
                "HF package file metadata requires a nonnegative byte size"
            )
        selected.append((filename, relative, size))
    if not any(relative == "manifest.json" for _, relative, _ in selected):
        raise ValueError("HF package subtree is missing manifest.json")
    total = sum(size for _, _, size in selected)
    if total > min(download.limit, unpacked.limit):
        raise ValueError("HF package byte limit exceeded")
    destination.mkdir()
    headers["Accept-Encoding"] = "identity"
    # Exact files are streamed directly, avoiding SDK symlinks and download metadata.
    # This also enforces actual transfer bounds when remote size metadata is wrong.
    with httpx.Client(follow_redirects=True, headers=headers) as client:
        for filename, relative, _ in sorted(selected):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            url = hf_hub_url(
                source.repo_id,
                filename,
                revision=source.revision,
                repo_type=source.repo_type,
            )
            _download(client, url, target, download, unpacked=unpacked)


def fetch_package(
    source: LocalPackageSource | HttpArchiveSource | HfPackageSource,
    cache_dir: Path,
    *,
    offline: bool = False,
    max_download_bytes: int = 1024 * 1024 * 1024,
    max_unpacked_bytes: int = 4 * 1024 * 1024 * 1024,
) -> ContextPackage:
    """Fetch a fixed package, validate it, and atomically publish its cache entry.

    Remote transport and authentication failures raise sanitized RuntimeError.

    Args:
        source: Existing local package, pinned HTTP ZIP, or pinned HF subtree.
        cache_dir: Directory containing source-key directories and their receipts.
        offline: Reuse a complete matching cache entry without network calls.
        max_download_bytes: Maximum actual bytes read from remote file responses.
        max_unpacked_bytes: Maximum total package bytes copied or unpacked.

    Returns:
        A verified package owned by the cache. Receipts are outside its root.

    Raises:
        ValueError: Invalid input, failed integrity checks, corrupt cache, or limits.
        FileNotFoundError: No complete matching cache is available offline.
    """
    for limit in (max_download_bytes, max_unpacked_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("package byte limits must be positive integers")
    if type(offline) is not bool:
        raise ValueError("offline must be a boolean")
    identity, local_package = _identity(source)
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    cache = _checked_absolute(cache_dir)
    if local_package is not None and (
        cache == local_package.root or local_package.root in cache.parents
    ):
        raise ValueError("package cache cannot be nested inside the local source")
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise ValueError("package cache must be a real directory")
    entry = cache / key
    if os.path.lexists(entry):
        return _read_cache(entry, identity)
    if offline:
        raise FileNotFoundError("no verified cache for this fixed package source")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=cache) as temporary:
        staging = Path(temporary)
        prepared = staging / "entry"
        prepared.mkdir()
        destination = prepared / "package"
        download = _ByteBudget(max_download_bytes)
        unpacked = _ByteBudget(max_unpacked_bytes)
        if local_package is not None:
            _copy_local(local_package, destination, unpacked)
        elif isinstance(source, HttpArchiveSource):
            _fetch_http(source, staging, destination, download, unpacked)
        else:
            _fetch_hf(source, destination, download, unpacked)
        package = ContextPackage.load(destination)
        _check_manifest(package, identity)
        receipt = {
            "schema_version": 1,
            "source": identity,
            "manifest_sha256": package.manifest_sha256,
        }
        (prepared / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        try:
            # A complete winner is nonempty: rename cannot replace it on POSIX or Windows.
            prepared.rename(entry)
        except OSError:
            if not os.path.lexists(entry):
                raise
            winner = _read_cache(entry, identity)
            if winner.manifest_sha256 != package.manifest_sha256:
                raise ValueError(
                    "concurrent downloads resolved to different package manifests"
                )
            return winner
        return _read_cache(entry, identity)
