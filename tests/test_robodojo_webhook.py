import hashlib
import hmac
import json
import time

import pytest

from robodojo.schemas import ArtifactPayload
from robodojo.publish.webhook import (
    DJANGO_SIGNATURE_HEADER,
    DJANGO_TIMESTAMP_HEADER,
    WebhookDeliveryError,
    build_django_finish_payload,
    canonical_json,
    notify_finish_webhook,
    post_finish_webhook,
    resolve_hmac_secret,
    sign_payload,
)

DJANGO_FINISH_URL = "https://api.test/api/v1/internal/eval/eval-1/trials/1/finish/"


def test_sign_payload_matches_hmac_sha256():
    body = b'{"status":"failed"}'
    secret = "test-secret"
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert sign_payload(body, secret) == f"sha256={expected}"


def test_post_finish_webhook_sends_django_signature_headers(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 200

    def fake_urlopen(request, timeout=30):
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        return FakeResponse()

    payload = build_django_finish_payload(
        status="failed",
        artifact=ArtifactPayload(bucket="b", prefix="evaluations/eval-1/"),
        metrics={"summary": {"trial_count": 1}},
        error={"code": "failed", "message": "policy down"},
    )

    post_finish_webhook(
        DJANGO_FINISH_URL,
        payload,
        secret="test-secret",
        opener=fake_urlopen,
        retry=False,
    )

    body = canonical_json(payload)
    headers = {name.lower(): value for name, value in captured["headers"].items()}
    assert headers[DJANGO_SIGNATURE_HEADER.lower()] == sign_payload(body, "test-secret")
    assert headers[DJANGO_TIMESTAMP_HEADER.lower()] == "1700000000"


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

    payload = build_django_finish_payload(
        status="failed",
        artifact=ArtifactPayload(bucket="b", prefix="evaluations/eval-1/"),
        metrics={"summary": {"trial_count": 1}},
    )

    with pytest.raises(WebhookDeliveryError, match="HTTP 503"):
        post_finish_webhook(
            DJANGO_FINISH_URL,
            payload,
            secret="test-secret",
            opener=lambda request, timeout=30: ErrorResponse(),
            retry=False,
        )


def test_post_finish_webhook_retries_transient_failures(monkeypatch):
    attempts = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 200

    class ErrorResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 503

    def fake_urlopen(request, timeout=30):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return ErrorResponse()
        return FakeResponse()

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    payload = build_django_finish_payload(
        status="failed",
        artifact=ArtifactPayload(bucket="b", prefix="evaluations/eval-1/"),
        metrics={"summary": {"failed": 1}},
        error={"code": "failed", "message": "policy down"},
    )

    result = post_finish_webhook(
        DJANGO_FINISH_URL,
        payload,
        secret="test-secret",
        opener=fake_urlopen,
    )

    assert result.status_code == 200
    assert attempts["count"] == 3
    assert sleeps == [1.0, 3.0]


def test_build_django_finish_payload_matches_control_plane():
    payload = build_django_finish_payload(
        status="done",
        artifact=ArtifactPayload(bucket="robodojo-artifacts", prefix="evaluations/e1/"),
        metrics={"summary": {"success_rate": 100.0}},
    )
    assert payload["status"] == "done"
    assert payload["score_inputs"] == {"success_rate": 100.0}
    assert payload["artifact"]["video_s3_key"] == "evaluations/e1/videos/main.mp4"


def test_build_django_finish_payload_includes_error():
    payload = build_django_finish_payload(
        status="failed",
        artifact=ArtifactPayload(bucket="b", prefix="evaluations/e1/"),
        metrics={"summary": {"trial_count": 1}},
        error={"code": "timeout", "message": "infer timed out"},
    )
    assert payload["error"] == {"code": "timeout", "message": "infer timed out"}


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
        status="done",
        finish_url=DJANGO_FINISH_URL,
        metrics={"summary": {"trial_count": 1}},
        artifact=ArtifactPayload(bucket="b", prefix="evaluations/eval-1/"),
        opener=fake_urlopen,
    )

    assert DJANGO_SIGNATURE_HEADER.lower() not in captured["headers"]
    assert captured["body"]["artifact"]["video_s3_key"].endswith("videos/main.mp4")


def test_notify_finish_webhook_includes_failed_error():
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
        status="failed",
        finish_url=DJANGO_FINISH_URL,
        metrics={"summary": {"failed": 1}},
        artifact=ArtifactPayload(bucket="b", prefix="evaluations/eval-1/"),
        error={"code": "failed", "message": "s3 unavailable"},
        secret="secret",
        opener=fake_urlopen,
    )

    assert result.status_code == 204
    assert captured["body"]["status"] == "failed"
    assert captured["body"]["error"] == {"code": "failed", "message": "s3 unavailable"}
