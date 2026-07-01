"""Upload evaluation artifacts to S3 (eval runner only; not used by Policy Server)."""

from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Callable

from eval_station.schemas import ArtifactPayload

UploadFileFn = Callable[[str, Path, str | None], None]

# Transient TOS/S3 failures (network blips, 5xx, throttling) are retried before
# the publish worker reports the upload as failed.
UPLOAD_RETRY_ATTEMPTS = 3
UPLOAD_RETRY_BACKOFF_S = (1.0, 3.0, 9.0)

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


def _guess_content_type(path: Path) -> str | None:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type


def _normalize_endpoint_url(raw: str) -> str:
    endpoint = raw.strip()
    if not endpoint:
        return ""
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return endpoint


def _s3_client_config(endpoint_url: str) -> Any:
    from botocore.config import Config

    style = _env_first(("S3_ADDRESSING_STYLE", "TOS_ADDRESSING_STYLE")).lower()
    if not style:
        style = "virtual" if endpoint_url else "auto"
    return Config(s3={"addressing_style": style})


def build_s3_client(
    s3_client: Any | None = None,
    *,
    endpoint_url: str | None = None,
    region_name: str | None = None,
) -> Any:
    """Build a boto3 S3 client, pointing at Volcano TOS when configured via env.

    TOS is S3-compatible: when TOS_ENDPOINT_URL / S3_ENDPOINT_URL (or AWS_ENDPOINT_URL)
    are set, boto3 targets TOS; otherwise it keeps default AWS behavior. Hostnames
    without a scheme are normalized to ``https://``. Credentials use the standard
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars.
    """
    if s3_client is not None:
        return s3_client

    import boto3

    resolved_endpoint = _normalize_endpoint_url(
        (endpoint_url or "").strip() or _env_first(_ENDPOINT_ENV_KEYS)
    )
    resolved_region = (region_name or "").strip() or _env_first(_REGION_ENV_KEYS) or None
    return boto3.client(
        "s3",
        endpoint_url=resolved_endpoint or None,
        region_name=resolved_region,
        config=_s3_client_config(resolved_endpoint),
    )


def upload_file_to_key(
    local_path: Path,
    *,
    bucket: str,
    key: str,
    s3_client: Any | None = None,
    upload_file: UploadFileFn | None = None,
    endpoint_url: str | None = None,
    region_name: str | None = None,
    retry: bool = True,
) -> str:
    """Upload a single local file to an explicit bucket/key (no prefix mangling).

    Transient upload errors are retried with backoff (see ``UPLOAD_RETRY_*``)
    so a brief TOS hiccup does not surface as a publish failure.
    """
    if not bucket:
        raise ValueError("bucket is required")
    if not local_path.is_file():
        raise FileNotFoundError(f"file not found for upload: {local_path}")

    client = build_s3_client(
        s3_client,
        endpoint_url=endpoint_url,
        region_name=region_name,
    )
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

    content_type = _guess_content_type(local_path)
    attempts = UPLOAD_RETRY_ATTEMPTS if retry else 1
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            uploader(key, local_path, content_type)
            return key
        except Exception as exc:  # noqa: BLE001 - retry then re-raise to caller
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(UPLOAD_RETRY_BACKOFF_S[attempt])
    assert last_err is not None
    raise last_err


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
