import hashlib
import hmac
import json

import pytest

from robodojo.schemas import ArtifactPayload
from robodojo.webhook import (
    FINISH_WEBHOOK_SCHEMA_VERSION,
    SIGNATURE_HEADER,
    WebhookDeliveryError,
    build_django_finish_payload,
    build_finish_payload,
    canonical_json,
    notify_finish_webhook,
    post_finish_webhook,
    resolve_hmac_secret,
    sign_payload,
)


def test_sign_payload_matches_hmac_sha256():
    body = b'{"evaluation_id":"eval-1","status":"planned"}'
    secret = "test-secret"
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert sign_payload(body, secret) == f"sha256={expected}"


def test_build_finish_payload_includes_required_fields():
    payload = build_finish_payload(
        evaluation_id="eval-1",
        status="planned",
        artifact_manifest_s3_key="eval-1/manifest.json",
        metrics={"summary": {"trial_count": 4}},
        error_summary="upload failed",
    )
    assert payload["schema_version"] == FINISH_WEBHOOK_SCHEMA_VERSION
    assert payload["artifact_manifest_s3_key"] == "eval-1/manifest.json"
    assert payload["error_summary"] == "upload failed"


def test_post_finish_webhook_sends_signature():
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 200

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        return FakeResponse()

    payload = build_finish_payload(
        evaluation_id="eval-1",
        status="planned",
        artifact_manifest_s3_key="eval-1/manifest.json",
        metrics={"summary": {"trial_count": 1}},
    )

    result = post_finish_webhook(
        "https://example.test/finish",
        payload,
        hmac_secret_ref="ROBODOJO_FINISH_HMAC_SECRET",
        secret="test-secret",
        opener=fake_urlopen,
    )

    body = canonical_json(payload)
    assert captured["body"] == body
    headers = {name.lower(): value for name, value in captured["headers"].items()}
    assert headers[SIGNATURE_HEADER.lower()] == sign_payload(body, "test-secret")
    assert result.status_code == 200


def test_resolve_hmac_secret_reads_environment(monkeypatch):
    monkeypatch.setenv("ROBODOJO_FINISH_HMAC_SECRET", "from-env")
    assert resolve_hmac_secret("ROBODOJO_FINISH_HMAC_SECRET") == "from-env"


def test_resolve_hmac_secret_empty_ref_returns_none():
    assert resolve_hmac_secret("") is None


def test_resolve_hmac_secret_missing_raises():
    with pytest.raises(KeyError, match="ROBODOJO_FINISH_HMAC_SECRET"):
        resolve_hmac_secret("ROBODOJO_FINISH_HMAC_SECRET")


def test_post_finish_webhook_raises_on_http_error_status():
    class ErrorResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 503

    payload = build_finish_payload(
        evaluation_id="eval-1",
        status="completed",
        artifact_manifest_s3_key="eval-1/manifest.json",
        metrics={"summary": {"trial_count": 1}},
    )

    with pytest.raises(WebhookDeliveryError, match="HTTP 503"):
        post_finish_webhook(
            "https://example.test/finish",
            payload,
            secret="test-secret",
            opener=lambda request, timeout=30: ErrorResponse(),
        )


def test_build_django_finish_payload_matches_control_plane():
    payload = build_django_finish_payload(
        status="done",
        result="success",
        artifact=ArtifactPayload(bucket="robodojo-artifacts", prefix="evaluations/e1/"),
        metrics={"summary": {"success_rate": 100.0, "latency_ms_avg": 12, "trial_count": 60}},
    )
    assert payload["status"] == "done"
    assert payload["artifact"]["video_s3_key"] == "evaluations/e1/videos/main.mp4"


def test_notify_finish_webhook_uses_django_body_without_signature():
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 200

    def fake_urlopen(request, timeout=30):
        captured["headers"] = {name.lower(): value for name, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    notify_finish_webhook(
        evaluation_id="eval-1",
        status="done",
        finish_url="https://api.test/api/v1/internal/eval/eval-1/trials/1/finish/",
        metrics={"summary": {"trial_count": 1}},
        artifact=ArtifactPayload(bucket="b", prefix="evaluations/eval-1/"),
        opener=fake_urlopen,
    )

    assert SIGNATURE_HEADER.lower() not in captured["headers"]
    assert captured["body"]["artifact"]["video_s3_key"].endswith("videos/main.mp4")


def test_notify_finish_webhook_wraps_builder():
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 204

    def fake_urlopen(request, timeout=30):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    result = notify_finish_webhook(
        evaluation_id="eval-1",
        status="failed",
        finish_url="https://example.test/finish",
        metrics={"summary": {"failed": 1}},
        artifact_manifest_s3_key=None,
        error_summary="s3 unavailable",
        secret="secret",
        opener=fake_urlopen,
    )

    assert result.status_code == 204
    assert captured["body"]["status"] == "failed"
    assert captured["body"]["error_summary"] == "s3 unavailable"
    assert "artifact_manifest_s3_key" not in captured["body"]
