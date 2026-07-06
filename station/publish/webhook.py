"""Finish webhook callback to the RoboDojo control plane (Django)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from station.schemas import ArtifactPayload


DJANGO_SIGNATURE_HEADER = "X-RoboDojo-Signature-256"
DJANGO_TIMESTAMP_HEADER = "X-RoboDojo-Signature-Timestamp"
WEBHOOK_RETRY_ATTEMPTS = 3
WEBHOOK_RETRY_BACKOFF_S = (1.0, 3.0, 9.0)
# Override scheme+host:port of the dispatched finish_url (path/query preserved),
# e.g. "http://192.168.101.71:8000". Empty -> use finish_url as-is.
FINISH_URL_BASE_ENV = "ROBODOJO_FINISH_URL_BASE"


def apply_finish_url_base(finish_url: str, base: str | None = None) -> str:
    if base is None:
        base = os.environ.get(FINISH_URL_BASE_ENV, "")
    base = (base or "").strip()
    if not base or not finish_url:
        return finish_url
    base_parts = urllib.parse.urlsplit(base if "//" in base else f"//{base}")
    url_parts = urllib.parse.urlsplit(finish_url)
    return urllib.parse.urlunsplit(
        (
            base_parts.scheme or url_parts.scheme,
            base_parts.netloc,
            url_parts.path,
            url_parts.query,
            url_parts.fragment,
        )
    )


class WebhookDeliveryError(RuntimeError):
    def __init__(self, finish_url: str, status_code: int, detail: str | None = None):
        message = f"finish webhook failed: {finish_url} -> HTTP {status_code}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)
        self.finish_url = finish_url
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class WebhookResult:
    finish_url: str
    status_code: int
    signature: str


def resolve_hmac_secret(secret_ref: str) -> str | None:
    if not secret_ref:
        return None
    value = os.environ.get(secret_ref)
    if value:
        return value
    raise KeyError(
        f"webhook HMAC secret not found in environment for ref '{secret_ref}'"
    )


def canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def sign_payload(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def build_django_finish_payload(
    *,
    status: str,
    artifact: ArtifactPayload,
    metrics: dict[str, Any],
    error: dict[str, Any] | None = None,
    video_key: str | None = None,
    hdf5_key: str | None = None,
    phase: str = "execution",
) -> dict[str, Any]:
    finish_status = (
        "done" if status in {"planned", "done", "success", "completed"} else "failed"
    )

    if phase == "publish":
        # The robot trial outcome was already persisted from the /start response.
        # This callback only patches the trial artifact with the TOS delivery keys
        # actually uploaded; missing keys mean that upload did not succeed.
        artifact_payload: dict[str, Any] = {
            "publish_status": "done" if finish_status == "done" else "failed",
        }
        if video_key:
            artifact_payload["video_s3_key"] = video_key
        if hdf5_key:
            artifact_payload["hdf5_s3_key"] = hdf5_key
        payload: dict[str, Any] = {
            "phase": "publish",
            "status": finish_status,
            "artifact": artifact_payload,
        }
        if error is not None:
            payload["error"] = error
        return payload

    prefix = artifact.prefix
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"

    raw_summary = metrics.get("summary")
    summary = raw_summary if isinstance(raw_summary, dict) else metrics

    trials = metrics.get("trials")
    trial_id = ""
    if isinstance(trials, list) and trials and isinstance(trials[0], dict):
        trial_id = str(trials[0].get("trial_id") or "")

    # `video_key` / `hdf5_key` (when provided) are the human-readable TOS delivery
    # keys, e.g. robodojo/{team}/{model}/{robot}/{task}/{eval_id}/trial_{index}.mp4
    # and .../trial_{index}.hdf5. The bookkeeping keys stay under the per-trial prefix.
    video_s3_key = video_key or f"{prefix}videos/{trial_id or 'main'}.mp4"

    artifact_payload = {
        "bucket": artifact.bucket,
        "prefix": prefix,
        "video_s3_key": video_s3_key,
        "manifest_key": f"{prefix}manifest.json",
        "metrics_key": f"{prefix}metrics.json",
        "events_key": f"{prefix}events.jsonl",
    }
    if hdf5_key:
        artifact_payload["hdf5_s3_key"] = hdf5_key
    payload = {
        "phase": "execution",
        "status": finish_status,
        "result": "success" if finish_status == "done" else "failed",
        "score_inputs": {
            "success_rate": summary.get("success_rate"),
        },
        "artifact": artifact_payload,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _signed_headers(body: bytes, secret: str) -> tuple[dict[str, str], str]:
    signature = sign_payload(body, secret)
    return (
        {
            DJANGO_SIGNATURE_HEADER: signature,
            DJANGO_TIMESTAMP_HEADER: str(int(time.time())),
        },
        signature,
    )


def _post_finish_webhook_once(
    finish_url: str,
    payload: dict[str, Any],
    *,
    hmac_secret_ref: str = "",
    secret: str | None = None,
    opener: Callable[..., Any] | None = None,
) -> WebhookResult:
    resolved_secret = secret
    if resolved_secret is None and hmac_secret_ref:
        resolved_secret = resolve_hmac_secret(hmac_secret_ref)
    body = canonical_json(payload)
    headers = {"Content-Type": "application/json"}
    signature = ""
    if resolved_secret:
        signature_headers, signature = _signed_headers(body, resolved_secret)
        headers.update(signature_headers)
    request = urllib.request.Request(
        finish_url,
        data=body,
        headers=headers,
        method="POST",
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=30) as response:
            status_code = int(
                getattr(response, "status", None) or response.getcode()
            )
    except urllib.error.HTTPError as exc:
        raise WebhookDeliveryError(
            finish_url,
            exc.code,
            detail=getattr(exc, "reason", None),
        ) from exc
    except urllib.error.URLError as exc:
        raise WebhookDeliveryError(
            finish_url,
            0,
            detail=str(exc.reason),
        ) from exc

    if status_code >= 400:
        raise WebhookDeliveryError(finish_url, status_code)

    return WebhookResult(
        finish_url=finish_url,
        status_code=status_code,
        signature=signature,
    )


def post_finish_webhook(
    finish_url: str,
    payload: dict[str, Any],
    *,
    hmac_secret_ref: str = "",
    secret: str | None = None,
    opener: Callable[..., Any] | None = None,
    retry: bool = True,
) -> WebhookResult:
    finish_url = apply_finish_url_base(finish_url)
    attempts = WEBHOOK_RETRY_ATTEMPTS if retry else 1
    last_err: WebhookDeliveryError | None = None
    for attempt in range(attempts):
        try:
            return _post_finish_webhook_once(
                finish_url,
                payload,
                hmac_secret_ref=hmac_secret_ref,
                secret=secret,
                opener=opener,
            )
        except WebhookDeliveryError as exc:
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(WEBHOOK_RETRY_BACKOFF_S[attempt])
    assert last_err is not None
    raise last_err


def notify_finish_webhook(
    *,
    status: str,
    finish_url: str,
    metrics: dict[str, Any],
    artifact: ArtifactPayload,
    hmac_secret_ref: str = "",
    error: dict[str, Any] | None = None,
    secret: str | None = None,
    opener: Callable[..., Any] | None = None,
    video_key: str | None = None,
    hdf5_key: str | None = None,
    phase: str = "execution",
) -> WebhookResult:
    payload = build_django_finish_payload(
        status=status,
        artifact=artifact,
        metrics=metrics,
        error=error,
        video_key=video_key,
        hdf5_key=hdf5_key,
        phase=phase,
    )
    return post_finish_webhook(
        finish_url,
        payload,
        hmac_secret_ref=hmac_secret_ref,
        secret=secret,
        opener=opener,
    )
