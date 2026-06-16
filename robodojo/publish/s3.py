"""Upload evaluation artifacts to S3 (eval runner only; not used by Policy Server)."""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from robodojo.publish.artifacts import MANIFEST_NAME
from robodojo.schemas import ArtifactPayload

UploadFileFn = Callable[[str, Path, str | None], None]

_ARTIFACT_BUCKET_ENV_KEYS = (
    "TOS_BUCKET",
    "S3_BUCKET",
    "AWS_S3_BUCKET",
)
_ARTIFACT_PREFIX_ENV_KEYS = (
    "TOS_PREFIX",
    "S3_PREFIX",
    "ROBODOJO_ARTIFACT_PREFIX",
)
_ENDPOINT_ENV_KEYS = (
    "TOS_ENDPOINT_URL",
    "S3_ENDPOINT_URL",
    "AWS_ENDPOINT_URL",
)
_REGION_ENV_KEYS = (
    "TOS_REGION",
    "S3_REGION",
    "AWS_REGION",
)


def _env_first(keys: tuple[str, ...]) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def normalize_endpoint_url(url: str) -> str:
    cleaned = url.strip()
    if cleaned and "://" not in cleaned:
        return f"https://{cleaned}"
    return cleaned


def resolve_artifact_payload(artifact: ArtifactPayload) -> ArtifactPayload:
    """Fill missing bucket/prefix from eval-station env vars when dispatch omits them."""
    bucket = artifact.bucket.strip() if artifact.bucket else ""
    if not bucket:
        bucket = _env_first(_ARTIFACT_BUCKET_ENV_KEYS)

    prefix = artifact.prefix.strip() if artifact.prefix else ""
    if not prefix:
        prefix = _env_first(_ARTIFACT_PREFIX_ENV_KEYS)

    if bucket == artifact.bucket and prefix == artifact.prefix:
        return artifact
    return artifact.model_copy(update={"bucket": bucket, "prefix": prefix})


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


def _s3_client_config() -> Any:
    from botocore.config import Config

    style = _env_first(("S3_ADDRESSING_STYLE", "TOS_ADDRESSING_STYLE")).lower()
    if not style:
        # Volcano TOS rejects path-style URLs ("Forbidden path to access server").
        style = "virtual" if _env_first(_ENDPOINT_ENV_KEYS) else "auto"
    return Config(s3={"addressing_style": style})


def build_s3_client(s3_client: Any | None = None) -> Any:
    """Build a boto3 S3 client, pointing at Volcano TOS when configured via env.

    TOS is S3-compatible: when TOS_ENDPOINT_URL / TOS_REGION (or the AWS_*
    equivalents) are set, boto3 targets TOS; otherwise it keeps default AWS
    behavior. Credentials use the standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.

    TOS requires virtual-hosted-style requests; see InvalidPathAccess /
    "Forbidden path to access server" when path-style is used.
    """
    if s3_client is not None:
        return s3_client

    import boto3

    endpoint_url = normalize_endpoint_url(_env_first(_ENDPOINT_ENV_KEYS))
    region_name = _env_first(_REGION_ENV_KEYS)
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        region_name=region_name or None,
        config=_s3_client_config(),
    )


def upload_file_to_key(
    local_path: Path,
    *,
    bucket: str,
    key: str,
    s3_client: Any | None = None,
    upload_file: UploadFileFn | None = None,
) -> str:
    """Upload a single local file to an explicit bucket/key (no prefix mangling)."""
    if not bucket:
        raise ValueError("bucket is required")
    if not local_path.is_file():
        raise FileNotFoundError(f"file not found for upload: {local_path}")

    client = build_s3_client(s3_client)
    uploader = upload_file
    if uploader is None:

        def _upload(upload_key: str, path: Path, content_type: str | None) -> None:
            _default_upload_file(
                client,
                bucket=bucket,
                key=upload_key,
                path=path,
                content_type=content_type,
            )

        uploader = _upload

    uploader(key, local_path, _guess_content_type(local_path))
    return key


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

    client = build_s3_client(s3_client)

    artifact = resolve_artifact_payload(artifact)
    if not artifact.bucket:
        raise ValueError(
            "artifact.bucket is required "
            "(set dispatch.artifact.bucket or TOS_BUCKET/S3_BUCKET env var)"
        )
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
