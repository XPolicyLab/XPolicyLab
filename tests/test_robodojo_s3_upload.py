from pathlib import Path

from robodojo.publish.artifacts import EVENTS_NAME, MANIFEST_NAME, METRICS_NAME, RUNNER_LOG_REL
from robodojo.publish.s3 import (
    artifact_s3_key,
    normalize_endpoint_url,
    normalize_s3_prefix,
    resolve_artifact_payload,
    upload_artifact_directory,
)
from robodojo.schemas import ArtifactPayload


def test_normalize_s3_prefix():
    assert normalize_s3_prefix("eval-1/") == "eval-1/"
    assert normalize_s3_prefix("eval-1") == "eval-1/"
    assert normalize_s3_prefix("") == ""


def test_normalize_endpoint_url_adds_https_scheme():
    assert normalize_endpoint_url("tos-s3-cn-shanghai.volces.com") == (
        "https://tos-s3-cn-shanghai.volces.com"
    )
    assert normalize_endpoint_url("https://example.test") == "https://example.test"


def test_resolve_artifact_payload_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "robodojo")
    monkeypatch.setenv("S3_PREFIX", "evaluations/eval-1/")

    resolved = resolve_artifact_payload(ArtifactPayload())

    assert resolved.bucket == "robodojo"
    assert resolved.prefix == "evaluations/eval-1/"


def test_upload_artifact_directory_uses_env_bucket_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "robodojo-artifacts")

    root = tmp_path / "artifacts"
    root.mkdir()
    (root / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    uploads: list[tuple[str, Path, str | None]] = []

    result = upload_artifact_directory(
        root,
        ArtifactPayload(prefix="eval-1/"),
        s3_client=object(),
        upload_file=lambda key, path, content_type: uploads.append(
            (key, path, content_type)
        ),
    )

    assert result.bucket == "robodojo-artifacts"
    assert result.manifest_s3_key == "eval-1/manifest.json"


def test_upload_artifact_directory_uses_prefix_and_manifest_last(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "videos").mkdir()
    (root / "logs").mkdir()
    (root / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    (root / METRICS_NAME).write_text("{}", encoding="utf-8")
    (root / EVENTS_NAME).write_text("{}\n", encoding="utf-8")
    (root / "logs" / "runner.log").write_text("ok", encoding="utf-8")
    (root / "videos" / "trial.mp4").write_bytes(b"")

    uploads: list[tuple[str, Path, str | None]] = []

    def fake_upload(key: str, path: Path, content_type: str | None) -> None:
        uploads.append((key, path, content_type))

    result = upload_artifact_directory(
        root,
        ArtifactPayload(bucket="robodojo-artifacts", prefix="eval-1/"),
        s3_client=object(),
        upload_file=fake_upload,
    )

    assert result.bucket == "robodojo-artifacts"
    assert result.prefix == "eval-1/"
    assert result.manifest_s3_key == "eval-1/manifest.json"
    assert uploads[-1][0] == "eval-1/manifest.json"
    assert len(uploads) == 5
    assert artifact_s3_key("eval-1/", "logs/runner.log") == "eval-1/logs/runner.log"
