"""Upload evaluation artifacts to S3 (eval runner only; not used by Policy Server)."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from robodojo.publish.artifacts import MANIFEST_NAME
from robodojo.schemas import ArtifactPayload

UploadFileFn = Callable[[str, Path, str | None], None]


def normalize_s3_prefix(prefix: str) -> str:
    cleaned = prefix.strip().strip("/")
    return f"{cleaned}/" if cleaned else ""


@dataclass(frozen=True)
class S3UploadResult:
    bucket: str
    prefix: str
    manifest_s3_key: str
    uploaded_keys: tuple[str, ...]


def artifact_s3_key(prefix: str, relative_path: str) -> str:
    return f"{normalize_s3_prefix(prefix)}{relative_path.replace(chr(92), '/')}"


def iter_artifact_files(root_dir: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root_dir).as_posix()
        files.append((path, relative))
    return files


def _guess_content_type(path: Path) -> str | None:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type


def _default_upload_file(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    path: Path,
    content_type: str | None,
) -> None:
    extra_args: dict[str, str] = {}
    if content_type:
        extra_args["ContentType"] = content_type
    if extra_args:
        s3_client.upload_file(str(path), bucket, key, ExtraArgs=extra_args)
    else:
        s3_client.upload_file(str(path), bucket, key)


def upload_artifact_directory(
    root_dir: Path,
    artifact: ArtifactPayload,
    *,
    s3_client: Any | None = None,
    upload_file: UploadFileFn | None = None,
) -> S3UploadResult:
    if not root_dir.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {root_dir}")

    client = s3_client
    if client is None:
        import boto3

        client = boto3.client("s3")

    if not artifact.bucket:
        raise ValueError("artifact.bucket is required")
    bucket = artifact.bucket
    prefix = normalize_s3_prefix(artifact.prefix)
    uploaded_keys: list[str] = []
    manifest_key = artifact_s3_key(prefix, MANIFEST_NAME)

    files = iter_artifact_files(root_dir)
    ordered = sorted(
        ((path, rel) for path, rel in files if rel != MANIFEST_NAME),
        key=lambda item: item[1],
    )
    ordered.extend((path, rel) for path, rel in files if rel == MANIFEST_NAME)

    uploader = upload_file
    if uploader is None:

        def _upload(key: str, path: Path, content_type: str | None) -> None:
            _default_upload_file(
                client,
                bucket=bucket,
                key=key,
                path=path,
                content_type=content_type,
            )

        uploader = _upload

    for path, relative in ordered:
        key = artifact_s3_key(prefix, relative)
        uploader(key, path, _guess_content_type(path))
        uploaded_keys.append(key)

    if manifest_key not in uploaded_keys:
        raise FileNotFoundError(f"{MANIFEST_NAME} was not uploaded from {root_dir}")

    return S3UploadResult(
        bucket=bucket,
        prefix=prefix,
        manifest_s3_key=manifest_key,
        uploaded_keys=tuple(uploaded_keys),
    )
