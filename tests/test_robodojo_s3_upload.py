from pathlib import Path
from unittest.mock import patch

from robodojo.publish.s3 import (
    build_s3_client,
    normalize_s3_prefix,
    resolve_artifact_payload,
    upload_file_to_key,
    _normalize_endpoint_url,
)
from robodojo.schemas import ArtifactPayload


def test_normalize_s3_prefix():
    assert normalize_s3_prefix("eval-1/") == "eval-1/"
    assert normalize_s3_prefix("eval-1") == "eval-1/"
    assert normalize_s3_prefix("") == ""


def test_upload_file_to_key_uploads_explicit_key(tmp_path):
    local = tmp_path / "trial_1.mp4"
    local.write_bytes(b"fake mp4")

    uploads: list[tuple[str, Path, str | None]] = []

    def fake_upload(key: str, path: Path, content_type: str | None) -> None:
        uploads.append((key, path, content_type))

    key = upload_file_to_key(
        local,
        bucket="robodojo-artifacts",
        key="evaluations/eval-1/trial_1.mp4",
        s3_client=object(),
        upload_file=fake_upload,
    )

    assert key == "evaluations/eval-1/trial_1.mp4"
    assert uploads == [
        ("evaluations/eval-1/trial_1.mp4", local, "video/mp4")
    ]


def test_upload_file_to_key_requires_existing_file(tmp_path):
    missing = tmp_path / "nope.mp4"

    try:
        upload_file_to_key(
            missing,
            bucket="robodojo-artifacts",
            key="evaluations/eval-1/trial_1.mp4",
            s3_client=object(),
            upload_file=lambda *args: None,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for missing file")


def test_upload_file_to_key_requires_bucket(tmp_path):
    local = tmp_path / "trial_1.mp4"
    local.write_bytes(b"fake mp4")

    try:
        upload_file_to_key(
            local,
            bucket="",
            key="evaluations/eval-1/trial_1.mp4",
            s3_client=object(),
            upload_file=lambda *args: None,
        )
    except ValueError as exc:
        assert str(exc) == "bucket is required"
    else:
        raise AssertionError("expected ValueError for missing bucket")


def test_resolve_artifact_payload_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "robodojo")
    monkeypatch.setenv("S3_PREFIX", "evaluations/eval-1/")

    resolved = resolve_artifact_payload(ArtifactPayload())

    assert resolved.bucket == "robodojo"
    assert resolved.prefix == "evaluations/eval-1/"


def test_resolve_artifact_payload_prefers_dispatch_values(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "env-bucket")

    resolved = resolve_artifact_payload(
        ArtifactPayload(bucket="dispatch-bucket", prefix="dispatch-prefix/")
    )

    assert resolved.bucket == "dispatch-bucket"
    assert resolved.prefix == "dispatch-prefix/"


def test_normalize_endpoint_url_adds_https_scheme():
    assert (
        _normalize_endpoint_url("tos-s3-cn-shanghai.volces.com")
        == "https://tos-s3-cn-shanghai.volces.com"
    )
    assert (
        _normalize_endpoint_url("https://tos-s3-cn-shanghai.volces.com")
        == "https://tos-s3-cn-shanghai.volces.com"
    )


def test_build_s3_client_uses_env_and_virtual_addressing(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "tos-s3-cn-shanghai.volces.com")
    monkeypatch.setenv("S3_REGION", "cn-shanghai")

    with patch("boto3.client") as mock_client:
        build_s3_client()

    mock_client.assert_called_once()
    args, kwargs = mock_client.call_args
    assert args == ("s3",)
    assert kwargs["endpoint_url"] == "https://tos-s3-cn-shanghai.volces.com"
    assert kwargs["region_name"] == "cn-shanghai"
    assert kwargs["config"].s3["addressing_style"] == "virtual"


def test_build_s3_client_prefers_explicit_endpoint_and_region():
    with patch("boto3.client") as mock_client:
        build_s3_client(
            endpoint_url="https://custom.example.com",
            region_name="us-east-1",
        )

    _, kwargs = mock_client.call_args
    assert kwargs["endpoint_url"] == "https://custom.example.com"
    assert kwargs["region_name"] == "us-east-1"
