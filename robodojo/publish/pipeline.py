"""Write evaluation artifacts and publish them to S3 / webhook."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

from robodojo.dispatch.errors import normalize_execution_error
from robodojo.dispatch.status import (
    STATUS_COMPLETED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PLANNED,
)
from robodojo.publish.artifacts import ArtifactWriter
from robodojo.publish.s3 import UploadFileFn, upload_artifact_directory
from robodojo.publish.webhook import WebhookDeliveryError, notify_finish_webhook
from robodojo.schemas import DispatchPayload


def _publish_exception_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [
        OSError,
        urllib.error.URLError,
        KeyError,
        WebhookDeliveryError,
        RuntimeError,
        ValueError,
        ConnectionError,
    ]
    try:
        from botocore.exceptions import BotoCoreError, ClientError

        types.extend([BotoCoreError, ClientError])
    except ImportError:
        pass
    return tuple(types)


PUBLISH_ERRORS = _publish_exception_types()


def write_artifacts(
    dispatch: DispatchPayload,
    trial_run: dict[str, object],
    artifact_dir: Path,
    *,
    evaluation_id: str,
    run_status: str = STATUS_PLANNED,
    policy_result: dict[str, Any] | None = None,
    error_summary: str | None = None,
) -> dict[str, str]:
    writer = ArtifactWriter(
        artifact_dir,
        evaluation_id=evaluation_id,
        dispatch=dispatch,
    )
    writer.setup()
    try:
        writer.emit_event("run_started")
        writer.register_trial(trial_run)
        trial_id = str(trial_run["trial_id"])
        if policy_result is not None:
            actions = policy_result.get("actions")
            action_count = len(actions) if isinstance(actions, list) else None
            writer.record_trial_start(trial_id)
            writer.record_trial_end(
                trial_id,
                status=STATUS_COMPLETED,
                metrics={
                    "success": True,
                    "steps": policy_result.get("steps", 1),
                    "action_count": action_count,
                },
            )
        writer.write_video_placeholder(trial_id)
        return writer.finalize(status=run_status, error_summary=error_summary)
    finally:
        writer.close()


def publish_artifacts(
    dispatch: DispatchPayload,
    artifact_paths: dict[str, str],
    *,
    run_status: str,
    upload_s3: bool,
    notify_webhook: bool,
    error: dict[str, Any] | None = None,
    finish_url: str = "",
    s3_client: Any | None = None,
    upload_file: UploadFileFn | None = None,
    webhook_secret: str | None = None,
    webhook_opener: Any | None = None,
) -> dict[str, Any]:
    published: dict[str, Any] = {}

    if upload_s3:
        upload_result = upload_artifact_directory(
            Path(artifact_paths["artifact_dir"]),
            dispatch.artifact,
            s3_client=s3_client,
            upload_file=upload_file,
        )
        published["s3"] = {
            "bucket": upload_result.bucket,
            "prefix": upload_result.prefix,
            "manifest_s3_key": upload_result.manifest_s3_key,
            "uploaded_count": len(upload_result.uploaded_keys),
        }

    if notify_webhook:
        metrics = json.loads(
            Path(artifact_paths["metrics"]).read_text(encoding="utf-8")
        )
        webhook_result = notify_finish_webhook(
            status=run_status,
            finish_url=finish_url,
            metrics=metrics,
            artifact=dispatch.artifact,
            hmac_secret_ref=dispatch.hmac_secret_ref,
            error=error,
            secret=webhook_secret,
            opener=webhook_opener,
        )
        published["webhook"] = {
            "finish_url": webhook_result.finish_url,
            "status_code": webhook_result.status_code,
        }

    return published


def publish_dispatch_artifacts(
    dispatch: DispatchPayload,
    artifact_paths: dict[str, str],
    *,
    run_status: str,
    upload_s3: bool,
    notify_webhook: bool,
    finish_url: str = "",
    error: dict[str, Any] | None = None,
    webhook_secret: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    published: dict[str, Any] = {}
    if run_status in {STATUS_DONE, STATUS_COMPLETED}:
        webhook_status = STATUS_COMPLETED
    elif run_status == STATUS_FAILED:
        webhook_status = STATUS_FAILED
    else:
        webhook_status = run_status
    try:
        if upload_s3:
            published.update(
                publish_artifacts(
                    dispatch,
                    artifact_paths,
                    run_status=webhook_status,
                    upload_s3=True,
                    notify_webhook=False,
                )
            )
        if notify_webhook:
            published.update(
                publish_artifacts(
                    dispatch,
                    artifact_paths,
                    run_status=webhook_status,
                    upload_s3=False,
                    notify_webhook=True,
                    finish_url=finish_url,
                    error=error,
                    webhook_secret=webhook_secret,
                )
            )
        return published, STATUS_COMPLETED, error
    except PUBLISH_ERRORS as exc:
        publish_error = normalize_execution_error(exc)
        failure_published: dict[str, Any] = {"error": publish_error["message"]}
        s3_info = published.get("s3")
        if isinstance(s3_info, dict) and s3_info.get("manifest_s3_key"):
            failure_published["s3"] = {
                "manifest_s3_key": str(s3_info["manifest_s3_key"])
            }
        if notify_webhook:
            try:
                failure_published.update(
                    publish_artifacts(
                        dispatch,
                        artifact_paths,
                        run_status=STATUS_FAILED,
                        upload_s3=False,
                        notify_webhook=True,
                        finish_url=finish_url,
                        error=publish_error,
                        webhook_secret=webhook_secret,
                    )
                )
            except PUBLISH_ERRORS as webhook_exc:
                failure_published["webhook_error"] = str(webhook_exc)
        return failure_published, STATUS_FAILED, publish_error
