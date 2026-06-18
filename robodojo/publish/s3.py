"""Upload evaluation artifacts to S3 (eval runner only; not used by Policy Server)."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Callable

UploadFileFn = Callable[[str, Path, str | None], None]


def normalize_s3_prefix(prefix: str) -> str:
    cleaned = prefix.strip().strip("/")
    return f"{cleaned}/" if cleaned else ""


def _guess_content_type(path: Path) -> str | None:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type


def build_s3_client(s3_client: Any | None = None) -> Any:
    """Build a boto3 S3 client, pointing at Volcano TOS when configured via env.

    TOS is S3-compatible: when TOS_ENDPOINT_URL / TOS_REGION (or the AWS_*
    equivalents) are set, boto3 targets TOS; otherwise it keeps default AWS
    behavior. Credentials use the standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
    """
    if s3_client is not None:
        return s3_client

    import boto3

    endpoint_url = os.environ.get("TOS_ENDPOINT_URL") or os.environ.get(
        "AWS_ENDPOINT_URL"
    )
    region_name = os.environ.get("TOS_REGION") or os.environ.get("AWS_REGION")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        region_name=region_name or None,
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
