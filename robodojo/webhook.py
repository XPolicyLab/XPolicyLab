"""Finish webhook callback to the RoboDojo control plane (Django)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from robodojo.schemas import ArtifactPayload


FINISH_WEBHOOK_SCHEMA_VERSION = "robodojo-finish-webhook-v1"
SIGNATURE_HEADER = "X-Robodojo-Signature"
DJANGO_FINISH_PATH_MARKER = "/internal/eval/"


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


def uses_django_finish_webhook(finish_url: str) -> bool:
    return DJANGO_FINISH_PATH_MARKER in finish_url


def canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def sign_payload(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def build_django_finish_payload(
    *,
    status: str,
    result: str | None,
    artifact: ArtifactPayload,
    metrics: dict[str, Any],
    error_summary: str | None = None,
) -> dict[str, Any]:
    prefix = artifact.prefix
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    summary = metrics.get("summary") if isinstance(metrics.get("summary"), dict) else metrics
    trials = metrics.get("trials") if isinstance(metrics.get("trials"), list) else []
    trial_id = ""
    if trials and isinstance(trials[0], dict):
        trial_id = str(trials[0].get("trial_id") or "")
    video_name = f"{trial_id}.mp4" if trial_id else "main.mp4"
    finish_status = "done" if status in {"planned", "done", "success", "completed"} else "failed"
    payload: dict[str, Any] = {
        "status": finish_status,
        "result": result or ("success" if finish_status == "done" else "failed"),
        "score_inputs": {
            "success_rate": summary.get("success_rate"),
            "latency_ms_avg": summary.get("latency_ms_avg"),
            "trial_count": summary.get("trial_count"),
            "completed_trial_count": summary.get("completed"),
        },
        "artifact": {
            "bucket": artifact.bucket,
            "prefix": prefix,
            "video_s3_key": f"{prefix}videos/{video_name}",
            "manifest_key": f"{prefix}manifest.json",
            "metrics_key": f"{prefix}metrics.json",
            "events_key": f"{prefix}events.jsonl",
        },
    }
    if error_summary:
        payload["error"] = {"code": status, "message": error_summary}
    return payload


def build_finish_payload(
    *,
    evaluation_id: str,
    status: str,
    artifact_manifest_s3_key: str | None,
    metrics: dict[str, Any],
    error_summary: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": FINISH_WEBHOOK_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "status": status,
        "metrics": metrics,
    }
    if artifact_manifest_s3_key is not None:
        payload["artifact_manifest_s3_key"] = artifact_manifest_s3_key
    if error_summary is not None:
        payload["error_summary"] = error_summary
    return payload


def post_finish_webhook(
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
        signature = sign_payload(body, resolved_secret)
        headers[SIGNATURE_HEADER] = signature
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


def notify_finish_webhook(
    *,
    evaluation_id: str,
    status: str,
    finish_url: str,
    hmac_secret_ref: str = "",
    metrics: dict[str, Any],
    artifact: ArtifactPayload | None = None,
    artifact_manifest_s3_key: str | None = None,
    error_summary: str | None = None,
    secret: str | None = None,
    opener: Callable[..., Any] | None = None,
) -> WebhookResult:
    if uses_django_finish_webhook(finish_url):
        if artifact is None:
            raise ValueError("artifact is required for Django finish webhook payloads")
        payload = build_django_finish_payload(
            status=status,
            result="success" if status != "failed" else None,
            artifact=artifact,
            metrics=metrics,
            error_summary=error_summary,
        )
    else:
        payload = build_finish_payload(
            evaluation_id=evaluation_id,
            status=status,
            artifact_manifest_s3_key=artifact_manifest_s3_key,
            metrics=metrics,
            error_summary=error_summary,
        )
    return post_finish_webhook(
        finish_url,
        payload,
        hmac_secret_ref=hmac_secret_ref,
        secret=secret,
        opener=opener,
    )
